use std::{
  collections::BTreeMap,
  io::{self, BufRead, Write},
  path::Path,
};

use netron_rs_formats::ToNormalizedJson;
use netron_rs_query::{EntityHandle, ModelSession, SessionLimits};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use super::{CLI_SCHEMA_VERSION, CliFailure, MappedModel};

#[derive(Deserialize)]
struct ServiceRequest {
  id: Option<Value>,
  method: String,
  #[serde(default)]
  params: Value,
}

struct ServiceSession {
  mapped: MappedModel,
  session: ModelSession,
}

#[derive(Default)]
struct ServiceState {
  next_session: u64,
  sessions: BTreeMap<u64, ServiceSession>,
}

pub(crate) fn serve_stdio() -> Result<(), CliFailure> {
  let stdin = io::stdin();
  let mut stdout = io::stdout().lock();
  let mut state = ServiceState::default();
  for line in stdin.lock().lines() {
    let line = line?;
    if line.trim().is_empty() {
      continue;
    }
    let response = state.handle_line(&line);
    serde_json::to_writer(&mut stdout, &response).map_err(CliFailure::from)?;
    stdout.write_all(b"\n")?;
    stdout.flush()?;
  }
  Ok(())
}

impl ServiceState {
  fn handle_line(&mut self, line: &str) -> Value {
    let request = match serde_json::from_str::<ServiceRequest>(line) {
      Ok(request) => request,
      Err(error) => {
        return error_response(
          None,
          None,
          CliFailure::invalid(format!("invalid JSON request: {error}")),
        );
      }
    };
    let id = request.id.clone();
    let method = request.method.clone();
    match self.handle(&request) {
      Ok(data) => success_response(id.as_ref(), &method, data),
      Err(error) => error_response(id.as_ref(), Some(&method), error.or_command("serve")),
    }
  }

  fn handle(&mut self, request: &ServiceRequest) -> Result<Value, CliFailure> {
    match request.method.as_str() {
      "open" => self.open(&request.params),
      "close" => self.close(&request.params),
      "summary" => Ok(
        serde_json::to_value(
          self
            .session(&request.params)?
            .summary(&limits_from_params(&request.params)?),
        )
        .map_err(CliFailure::from)?,
      ),
      "diagnostics" => Ok(
        serde_json::to_value(
          self
            .session(&request.params)?
            .diagnostics(&limits_from_params(&request.params)?),
        )
        .map_err(CliFailure::from)?,
      ),
      "search" => self.search(&request.params),
      "detail" => self.detail(&request.params),
      "slice" => self.slice(&request.params),
      "layout" => self.layout(&request.params),
      "export" => self.export(&request.params),
      "mlir.symbols" => self.mlir_symbols(&request.params),
      "onnx.tensor" => self.onnx_tensor(&request.params),
      method => Err(CliFailure::invalid(format!(
        "unknown service method '{method}'"
      ))),
    }
  }

  fn open(&mut self, params: &Value) -> Result<Value, CliFailure> {
    let path = string_param(params, "path")?;
    let mapped = MappedModel::open(Path::new(path))?;
    let session = ModelSession::open(mapped.bytes(), mapped.source()).map_err(CliFailure::from)?;
    let summary = session.summary(&limits_from_params(params)?);
    self.next_session += 1;
    let session_id = self.next_session;
    self
      .sessions
      .insert(session_id, ServiceSession { mapped, session });
    Ok(json!({ "session": session_id, "summary": summary }))
  }

  fn close(&mut self, params: &Value) -> Result<Value, CliFailure> {
    let session = session_id_param(params)?;
    let closed = self.sessions.remove(&session).is_some();
    Ok(json!({ "session": session, "closed": closed }))
  }

  fn search(&self, params: &Value) -> Result<Value, CliFailure> {
    let query = string_param(params, "query")?;
    let mut limits = limits_from_params(params)?;
    if let Some(limit) = optional_usize_param(params, "limit")? {
      limits.search = limit;
    }
    serde_json::to_value(self.session(params)?.search(query, &limits)).map_err(CliFailure::from)
  }

  fn detail(&self, params: &Value) -> Result<Value, CliFailure> {
    let handle = handle_param(params)?;
    let detail = self
      .session(params)?
      .detail(&handle, &limits_from_params(params)?)
      .ok_or_else(|| CliFailure::invalid("detail handle is not available"))?;
    serde_json::to_value(detail).map_err(CliFailure::from)
  }

  fn slice(&self, params: &Value) -> Result<Value, CliFailure> {
    let handle = handle_param(params)?;
    let slice = self
      .session(params)?
      .slice(&handle, &projection_limits_from_params(params)?)
      .ok_or_else(|| CliFailure::invalid("slice scope is not available"))?;
    serde_json::to_value(slice).map_err(CliFailure::from)
  }

  fn layout(&self, params: &Value) -> Result<Value, CliFailure> {
    let handle = handle_param(params)?;
    let layout = self
      .session(params)?
      .layout(&handle, &projection_limits_from_params(params)?)
      .ok_or_else(|| CliFailure::invalid("layout scope is not available"))?;
    serde_json::to_value(layout).map_err(CliFailure::from)
  }

  fn export(&self, params: &Value) -> Result<Value, CliFailure> {
    let entry = self.entry(params)?;
    let limits = limits_from_params(params)?.clamp();
    let model = netron_rs_formats::parse(entry.mapped.input()).map_err(CliFailure::from)?;
    let mut normalized =
      serde_json::from_str::<Value>(&model.to_normalized_json().map_err(CliFailure::from)?)
        .map_err(CliFailure::from)?;
    let omitted_count = truncate_json_arrays(&mut normalized, limits.export);
    Ok(json!({
        "session": session_id_param(params)?,
        "format": entry.session.summary(&limits).format,
        "limit_used": limits.export,
        "truncated": omitted_count > 0,
        "omitted_count": omitted_count,
        "normalized": normalized
    }))
  }

  fn mlir_symbols(&self, params: &Value) -> Result<Value, CliFailure> {
    let symbols = self
      .session(params)?
      .mlir_symbols(&limits_from_params(params)?)
      .ok_or_else(|| CliFailure::invalid("MLIR symbols require an MLIR model"))?;
    serde_json::to_value(symbols).map_err(CliFailure::from)
  }

  fn onnx_tensor(&self, params: &Value) -> Result<Value, CliFailure> {
    let tensor = usize_param(params, "tensor")?;
    let metadata = self
      .session(params)?
      .tensor_metadata(&limits_from_params(params)?)
      .into_iter()
      .find(|entry| matches!(entry.handle, EntityHandle::Tensor { tensor: id } if id == tensor))
      .ok_or_else(|| CliFailure::invalid("tensor id is not available"))?;
    serde_json::to_value(metadata).map_err(CliFailure::from)
  }

  fn session(&self, params: &Value) -> Result<&ModelSession, CliFailure> {
    Ok(&self.entry(params)?.session)
  }

  fn entry(&self, params: &Value) -> Result<&ServiceSession, CliFailure> {
    let id = session_id_param(params)?;
    self
      .sessions
      .get(&id)
      .ok_or_else(|| CliFailure::invalid(format!("session {id} is not open")))
  }
}

fn success_response(id: Option<&Value>, command: &str, data: Value) -> Value {
  let mut response = Map::new();
  response.insert("schema_version".to_owned(), json!(CLI_SCHEMA_VERSION));
  response.insert("status".to_owned(), json!("ok"));
  response.insert("command".to_owned(), json!(command));
  if let Some(id) = id {
    response.insert("id".to_owned(), id.clone());
  }
  response.insert("data".to_owned(), data);
  Value::Object(response)
}

fn error_response(id: Option<&Value>, command: Option<&str>, error: CliFailure) -> Value {
  let mut response = Map::new();
  response.insert("schema_version".to_owned(), json!(CLI_SCHEMA_VERSION));
  response.insert("status".to_owned(), json!("error"));
  response.insert(
    "command".to_owned(),
    command.map_or(Value::Null, |command| json!(command)),
  );
  if let Some(id) = id {
    response.insert("id".to_owned(), id.clone());
  }
  response.insert(
    "error".to_owned(),
    json!({ "code": error.kind, "message": error.message }),
  );
  Value::Object(response)
}

fn session_id_param(params: &Value) -> Result<u64, CliFailure> {
  params
    .get("session")
    .and_then(Value::as_u64)
    .ok_or_else(|| CliFailure::invalid("session must be an unsigned integer"))
}

fn string_param<'a>(params: &'a Value, name: &str) -> Result<&'a str, CliFailure> {
  params
    .get(name)
    .and_then(Value::as_str)
    .ok_or_else(|| CliFailure::invalid(format!("{name} must be a string")))
}

fn usize_param(params: &Value, name: &str) -> Result<usize, CliFailure> {
  value_to_usize(
    params
      .get(name)
      .ok_or_else(|| CliFailure::invalid(format!("{name} is required")))?,
    name,
  )
}

fn optional_usize_param(params: &Value, name: &str) -> Result<Option<usize>, CliFailure> {
  params
    .get(name)
    .map(|value| value_to_usize(value, name))
    .transpose()
}

fn value_to_usize(value: &Value, name: &str) -> Result<usize, CliFailure> {
  value
    .as_u64()
    .and_then(|value| usize::try_from(value).ok())
    .ok_or_else(|| CliFailure::invalid(format!("{name} must be an unsigned integer")))
}

fn limits_from_params(params: &Value) -> Result<SessionLimits, CliFailure> {
  let mut limits = SessionLimits::default();
  if let Some(value) = optional_usize_param(params, "limit")? {
    limits.export = value;
    limits.search = value;
    limits.slice = value;
    limits.layout = value;
    limits.preview = value;
    limits.detail = value;
    limits.diagnostics = value;
  }
  if let Some(object) = params.get("limits") {
    for (key, value) in object
      .as_object()
      .ok_or_else(|| CliFailure::invalid("limits must be an object"))?
    {
      let value = value_to_usize(value, key)?;
      match key.as_str() {
        "export" => limits.export = value,
        "search" => limits.search = value,
        "slice" => limits.slice = value,
        "layout" => limits.layout = value,
        "preview" => limits.preview = value,
        "detail" => limits.detail = value,
        "diagnostics" => limits.diagnostics = value,
        _ => return Err(CliFailure::invalid(format!("unknown limit '{key}'"))),
      }
    }
  }
  Ok(limits)
}

fn projection_limits_from_params(params: &Value) -> Result<SessionLimits, CliFailure> {
  let mut limits = limits_from_params(params)?;
  if let Some(value) = optional_usize_param(params, "max_nodes")? {
    limits.slice = value;
    limits.layout = value;
  }
  if let Some(value) = optional_usize_param(params, "max_ops")? {
    limits.slice = value;
    limits.layout = value;
  }
  Ok(limits)
}

fn handle_param(params: &Value) -> Result<EntityHandle, CliFailure> {
  parse_handle(
    params
      .get("handle")
      .ok_or_else(|| CliFailure::invalid("handle is required"))?,
  )
}

fn parse_handle(handle: &Value) -> Result<EntityHandle, CliFailure> {
  let kind = string_param(handle, "kind")?;
  match kind {
    "graph" => Ok(EntityHandle::Graph {
      graph: usize_param(handle, "graph")?,
    }),
    "node" => Ok(EntityHandle::Node {
      graph: usize_param(handle, "graph")?,
      node: usize_param(handle, "node")?,
    }),
    "value" => Ok(EntityHandle::Value {
      graph: usize_param(handle, "graph")?,
      value: usize_param(handle, "value")?,
    }),
    "tensor" => Ok(EntityHandle::Tensor {
      tensor: usize_param(handle, "tensor")?,
    }),
    "function" => Ok(EntityHandle::Function {
      function: usize_param(handle, "function")?,
    }),
    "metadata" => Ok(EntityHandle::Metadata {
      owner: string_param(handle, "owner")?.to_owned(),
      key: string_param(handle, "key")?.to_owned(),
    }),
    "operator_set" => Ok(EntityHandle::OperatorSet {
      domain: handle
        .get("domain")
        .and_then(Value::as_str)
        .map(str::to_owned),
      version: handle
        .get("version")
        .and_then(Value::as_i64)
        .ok_or_else(|| CliFailure::invalid("version must be an integer"))?,
    }),
    "diagnostic" => Ok(EntityHandle::Diagnostic {
      diagnostic: usize_param(handle, "diagnostic")?,
    }),
    "mlir_module" => Ok(EntityHandle::MlirModule {
      module: usize_param(handle, "module")?,
    }),
    "mlir_function" => Ok(EntityHandle::MlirFunction {
      function: usize_param(handle, "function")?,
    }),
    "mlir_operation" => Ok(EntityHandle::MlirOperation {
      scope: string_param(handle, "scope")?.to_owned(),
      operation: usize_param(handle, "operation")?,
    }),
    "mlir_value" => Ok(EntityHandle::MlirValue {
      scope: string_param(handle, "scope")?.to_owned(),
      value: usize_param(handle, "value")?,
    }),
    "mlir_region" => Ok(EntityHandle::MlirRegion {
      scope: string_param(handle, "scope")?.to_owned(),
      region: usize_param(handle, "region")?,
    }),
    "mlir_block" => Ok(EntityHandle::MlirBlock {
      scope: string_param(handle, "scope")?.to_owned(),
      block: usize_param(handle, "block")?,
    }),
    "mlir_symbol" => Ok(EntityHandle::MlirSymbol {
      symbol: usize_param(handle, "symbol")?,
    }),
    "mlir_dialect" => Ok(EntityHandle::MlirDialect {
      dialect: string_param(handle, "dialect")?.to_owned(),
    }),
    "mlir_attribute" => Ok(EntityHandle::MlirAttribute {
      scope: string_param(handle, "scope")?.to_owned(),
      attribute: usize_param(handle, "attribute")?,
    }),
    "mlir_resource" => Ok(EntityHandle::MlirResource {
      resource: usize_param(handle, "resource")?,
    }),
    _ => Err(CliFailure::invalid(format!("unknown handle kind '{kind}'"))),
  }
}

fn truncate_json_arrays(value: &mut Value, limit: usize) -> usize {
  match value {
    Value::Array(items) => {
      let mut omitted = items.len().saturating_sub(limit);
      items.truncate(limit);
      omitted += items
        .iter_mut()
        .map(|item| truncate_json_arrays(item, limit))
        .sum::<usize>();
      omitted
    }
    Value::Object(object) => object
      .values_mut()
      .map(|value| truncate_json_arrays(value, limit))
      .sum(),
    _ => 0,
  }
}
