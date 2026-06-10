use std::{
  collections::BTreeMap,
  fs::File,
  io::{self, BufRead, BufReader, Read, Write},
  net::{TcpListener, TcpStream},
  path::Path,
  sync::{
    Arc, Mutex, MutexGuard,
    atomic::{AtomicU64, Ordering},
    mpsc,
  },
  thread,
  time::{SystemTime, UNIX_EPOCH},
};

use netron_rs_layout::CancellationToken;
use netron_rs_query::{
  CollapseMode, DEFAULT_NODE_SLICE_DEPTH, EntityHandle, ModelSession, ProjectionError,
  ProjectionOptions, SessionLimits,
};
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
  session: ModelSession,
}

type SharedServiceSession = Arc<Mutex<ServiceSession>>;
type ServiceSessionMap = BTreeMap<u64, SharedServiceSession>;
type CancellationMap = BTreeMap<String, CancellationToken>;

#[derive(Default)]
struct ServiceState {
  next_session: AtomicU64,
  sessions: Mutex<ServiceSessionMap>,
  cancellations: Mutex<CancellationMap>,
}

pub(crate) fn serve_stdio() -> Result<(), CliFailure> {
  let stdin = io::stdin();
  let state = Arc::new(ServiceState::default());
  let (responses, writer) = mpsc::channel::<Value>();
  let writer = thread::spawn(move || -> Result<(), CliFailure> {
    let mut stdout = io::stdout().lock();
    for response in writer {
      serde_json::to_writer(&mut stdout, &response).map_err(CliFailure::from)?;
      stdout.write_all(b"\n")?;
      stdout.flush()?;
    }
    Ok(())
  });
  for line in stdin.lock().lines() {
    let line = line?;
    if line.trim().is_empty() {
      continue;
    }
    let state = Arc::clone(&state);
    let responses = responses.clone();
    let _ = thread::spawn(move || {
      let response = state.handle_line(&line);
      let _ = responses.send(response);
    });
  }
  drop(responses);
  writer
    .join()
    .map_err(|_| CliFailure::internal("stdio service writer panicked"))?
}

pub(crate) fn serve_http(listen: &str) -> Result<(), CliFailure> {
  let listener = TcpListener::bind(listen)
    .map_err(|error| CliFailure::invalid(format!("failed to bind {listen}: {error}")))?;
  let address = listener.local_addr().map_err(CliFailure::from)?;
  let token = http_token();
  println!(
    "{}",
    json!({
      "schema_version": CLI_SCHEMA_VERSION,
      "status": "ok",
      "command": "serve",
      "data": {
        "transport": "http",
        "address": address.to_string(),
        "token": token,
      }
    })
  );
  io::stdout().flush()?;

  let state = Arc::new(ServiceState::default());
  for stream in listener.incoming() {
    match stream {
      Ok(stream) => {
        let state = Arc::clone(&state);
        let token = token.clone();
        thread::spawn(move || {
          let mut stream = stream;
          let response = handle_http_connection(&state, &token, &mut stream);
          if let Err(error) = response {
            let body = error_response(None, Some("serve"), error.or_command("serve"));
            let _ = write_http_json(&mut stream, 400, &body);
          }
        });
      }
      Err(error) => return Err(CliFailure::from(error)),
    }
  }
  Ok(())
}

impl ServiceState {
  fn handle_line(&self, line: &str) -> Value {
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
    let registration = match self.register_request_cancellation(&request) {
      Ok(registration) => registration,
      Err(error) => {
        return error_response(id.as_ref(), Some(&method), error.or_command("serve"));
      }
    };
    let cancel = registration.as_ref().map(|(_, cancel)| cancel);
    let result = self.handle(&request, cancel);
    if let Some((key, _)) = registration {
      let _ = self.unregister_cancellation(&key);
    }
    match result {
      Ok(data) => success_response(id.as_ref(), &method, data),
      Err(error) => error_response(id.as_ref(), Some(&method), error.or_command("serve")),
    }
  }

  fn handle(
    &self,
    request: &ServiceRequest,
    cancel: Option<&CancellationToken>,
  ) -> Result<Value, CliFailure> {
    match request.method.as_str() {
      "open" => self.open(&request.params),
      "close" => self.close(&request.params),
      "cancel" => self.cancel(&request.params),
      "summary" => self.summary(&request.params),
      "diagnostics" => self.diagnostics(&request.params),
      "search" => self.search(&request.params),
      "detail" => self.detail(&request.params),
      "slice" => self.slice(&request.params, cancel),
      "layout" => self.layout(&request.params, cancel),
      "export" => self.export(&request.params),
      "mlir.symbols" => self.mlir_symbols(&request.params),
      "mlir.symbol.tree" => self.mlir_symbol_tree(&request.params),
      "onnx.tensor" => self.onnx_tensor(&request.params),
      method => Err(CliFailure::invalid(format!(
        "unknown service method '{method}'"
      ))),
    }
  }

  fn open(&self, params: &Value) -> Result<Value, CliFailure> {
    let path = string_param(params, "path")?;
    let mapped = MappedModel::open(Path::new(path))?;
    let session = ModelSession::open(mapped.bytes(), mapped.source()).map_err(CliFailure::from)?;
    let summary = session.summary(&limits_from_params(params)?);
    let session_id = self.next_session.fetch_add(1, Ordering::Relaxed) + 1;
    self
      .lock_sessions()?
      .insert(session_id, Arc::new(Mutex::new(ServiceSession { session })));
    Ok(json!({ "session": session_id, "summary": summary }))
  }

  fn close(&self, params: &Value) -> Result<Value, CliFailure> {
    let session = session_id_param(params)?;
    let closed = self.lock_sessions()?.remove(&session).is_some();
    Ok(json!({ "session": session, "closed": closed }))
  }

  fn cancel(&self, params: &Value) -> Result<Value, CliFailure> {
    let cancel_token = cancel_token_param(params)?;
    let canceled = self
      .lock_cancellations()?
      .get(&cancel_token)
      .is_some_and(|token| {
        token.cancel();
        true
      });
    let mut data = Map::new();
    data.insert("cancel_token".to_owned(), json!(cancel_token));
    data.insert("canceled".to_owned(), json!(canceled));
    if let Some(session) = params.get("session").and_then(Value::as_u64) {
      data.insert("session".to_owned(), json!(session));
    }
    Ok(Value::Object(data))
  }

  fn summary(&self, params: &Value) -> Result<Value, CliFailure> {
    let entry = self.entry(params)?;
    let entry = lock_service_session(&entry)?;
    serde_json::to_value(entry.session.summary(&limits_from_params(params)?))
      .map_err(CliFailure::from)
  }

  fn diagnostics(&self, params: &Value) -> Result<Value, CliFailure> {
    let entry = self.entry(params)?;
    let entry = lock_service_session(&entry)?;
    serde_json::to_value(entry.session.diagnostics(&limits_from_params(params)?))
      .map_err(CliFailure::from)
  }

  fn search(&self, params: &Value) -> Result<Value, CliFailure> {
    let query = string_param(params, "query")?;
    let mut limits = limits_from_params(params)?;
    if let Some(limit) = optional_usize_param(params, "limit")? {
      limits.search = limit;
    }
    let entry = self.entry(params)?;
    let entry = lock_service_session(&entry)?;
    if params.get("cursor").is_some() {
      return serde_json::to_value(entry.session.search_page(
        query,
        optional_cursor_param(params)?,
        &limits,
      ))
      .map_err(CliFailure::from);
    }
    serde_json::to_value(entry.session.search(query, &limits)).map_err(CliFailure::from)
  }

  fn detail(&self, params: &Value) -> Result<Value, CliFailure> {
    let handle = handle_param(params)?;
    let entry = self.entry(params)?;
    let entry = lock_service_session(&entry)?;
    let detail = entry
      .session
      .detail(&handle, &limits_from_params(params)?)
      .ok_or_else(|| CliFailure::invalid("detail handle is not available"))?;
    serde_json::to_value(detail).map_err(CliFailure::from)
  }

  fn slice(&self, params: &Value, cancel: Option<&CancellationToken>) -> Result<Value, CliFailure> {
    let handle = handle_param(params)?;
    let depth = optional_usize_param(params, "depth")?.unwrap_or(DEFAULT_NODE_SLICE_DEPTH);
    let collapse = collapse_param(params)?;
    let entry = self.entry(params)?;
    let entry = lock_service_session(&entry)?;
    let slice = match entry.session.try_slice_with_options(
      &handle,
      &projection_limits_from_params(params)?,
      ProjectionOptions {
        collapse,
        node_depth: depth,
        cancel: cancel.cloned(),
      },
    ) {
      Ok(Some(slice)) => slice,
      Ok(None) => return Err(CliFailure::invalid("slice scope is not available")),
      Err(ProjectionError::Canceled) => return Err(CliFailure::canceled("request canceled")),
    };
    serde_json::to_value(slice).map_err(CliFailure::from)
  }

  fn layout(
    &self,
    params: &Value,
    cancel: Option<&CancellationToken>,
  ) -> Result<Value, CliFailure> {
    let handle = handle_param(params)?;
    let collapse = collapse_param(params)?;
    let entry = self.entry(params)?;
    let entry = lock_service_session(&entry)?;
    let layout = match entry.session.try_layout_with_options(
      &handle,
      &projection_limits_from_params(params)?,
      ProjectionOptions {
        collapse,
        cancel: cancel.cloned(),
        ..ProjectionOptions::default()
      },
    ) {
      Ok(Some(layout)) => layout,
      Ok(None) => return Err(CliFailure::invalid("layout scope is not available")),
      Err(ProjectionError::Canceled) => return Err(CliFailure::canceled("request canceled")),
    };
    serde_json::to_value(layout).map_err(CliFailure::from)
  }

  fn export(&self, params: &Value) -> Result<Value, CliFailure> {
    let session = session_id_param(params)?;
    let entry = self.entry(params)?;
    let entry = lock_service_session(&entry)?;
    let limits = limits_from_params(params)?.clamp();
    let mut export =
      serde_json::to_value(entry.session.export(&limits)).map_err(CliFailure::from)?;
    if let Value::Object(data) = &mut export {
      data.insert("session".to_owned(), json!(session));
    }
    Ok(export)
  }

  fn mlir_symbols(&self, params: &Value) -> Result<Value, CliFailure> {
    let entry = self.entry(params)?;
    let entry = lock_service_session(&entry)?;
    let symbols = entry
      .session
      .mlir_symbols(&limits_from_params(params)?)
      .ok_or_else(|| CliFailure::invalid("MLIR symbols require an MLIR model"))?;
    serde_json::to_value(symbols).map_err(CliFailure::from)
  }

  fn mlir_symbol_tree(&self, params: &Value) -> Result<Value, CliFailure> {
    let entry = self.entry(params)?;
    let entry = lock_service_session(&entry)?;
    let tree = entry
      .session
      .mlir_symbol_tree(&limits_from_params(params)?)
      .ok_or_else(|| CliFailure::invalid("MLIR symbol tree requires an MLIR model"))?;
    serde_json::to_value(tree).map_err(CliFailure::from)
  }

  fn onnx_tensor(&self, params: &Value) -> Result<Value, CliFailure> {
    let tensor = usize_param(params, "tensor")?;
    let entry = self.entry(params)?;
    let entry = lock_service_session(&entry)?;
    let metadata = entry
      .session
      .tensor_metadata_by_id(tensor)
      .ok_or_else(|| CliFailure::invalid("tensor id is not available"))?;
    serde_json::to_value(metadata).map_err(CliFailure::from)
  }

  fn entry(&self, params: &Value) -> Result<SharedServiceSession, CliFailure> {
    let id = session_id_param(params)?;
    self
      .lock_sessions()?
      .get(&id)
      .cloned()
      .ok_or_else(|| CliFailure::invalid(format!("session {id} is not open")))
  }

  fn lock_sessions(&self) -> Result<MutexGuard<'_, ServiceSessionMap>, CliFailure> {
    self
      .sessions
      .lock()
      .map_err(|_| CliFailure::internal("service session map lock poisoned"))
  }

  fn lock_cancellations(&self) -> Result<MutexGuard<'_, CancellationMap>, CliFailure> {
    self
      .cancellations
      .lock()
      .map_err(|_| CliFailure::internal("service cancellation map lock poisoned"))
  }

  fn register_request_cancellation(
    &self,
    request: &ServiceRequest,
  ) -> Result<Option<(String, CancellationToken)>, CliFailure> {
    if request.method == "cancel" {
      return Ok(None);
    }
    if !matches!(request.method.as_str(), "layout" | "slice") {
      return Ok(None);
    }
    let Some(key) = request_cancellation_key(request)? else {
      return Ok(None);
    };
    let cancel = CancellationToken::default();
    let mut cancellations = self.lock_cancellations()?;
    if cancellations.contains_key(&key) {
      return Err(CliFailure::invalid("cancel_token is already active"));
    }
    cancellations.insert(key.clone(), cancel.clone());
    Ok(Some((key, cancel)))
  }

  fn unregister_cancellation(&self, key: &str) -> Result<(), CliFailure> {
    self.lock_cancellations()?.remove(key);
    Ok(())
  }
}

fn lock_service_session(
  entry: &SharedServiceSession,
) -> Result<MutexGuard<'_, ServiceSession>, CliFailure> {
  entry
    .lock()
    .map_err(|_| CliFailure::internal("service session lock poisoned"))
}

struct HttpRequest {
  method: String,
  path: String,
  authorization: Option<String>,
  body: String,
}

fn handle_http_connection(
  state: &ServiceState,
  token: &str,
  stream: &mut TcpStream,
) -> Result<(), CliFailure> {
  let request = read_http_request(stream)?;
  if request.method == "OPTIONS" {
    return write_http_empty(stream, 204);
  }
  if request.method == "GET" && request.path == "/health" {
    return write_http_json(
      stream,
      200,
      &json!({
        "schema_version": CLI_SCHEMA_VERSION,
        "status": "ok",
        "command": "serve",
        "data": { "transport": "http" }
      }),
    );
  }
  if request.method != "POST" {
    let response = error_response(
      None,
      Some("serve"),
      CliFailure::invalid("HTTP service accepts POST requests only").or_command("serve"),
    );
    return write_http_json(stream, 405, &response);
  }
  let expected_token = format!("Bearer {token}");
  if request.authorization.as_deref() != Some(expected_token.as_str()) {
    let response = error_response(
      None,
      Some("serve"),
      CliFailure::access_denied("missing or invalid bearer token").or_command("serve"),
    );
    return write_http_json(stream, 401, &response);
  }
  let response = state.handle_line(&request.body);
  write_http_json(stream, 200, &response)
}

fn read_http_request(stream: &mut TcpStream) -> Result<HttpRequest, CliFailure> {
  let mut reader = BufReader::new(stream);
  let mut request_line = String::new();
  reader.read_line(&mut request_line)?;
  let mut parts = request_line.split_whitespace();
  let method = parts
    .next()
    .ok_or_else(|| CliFailure::invalid("missing HTTP method"))?
    .to_owned();
  let path = parts
    .next()
    .ok_or_else(|| CliFailure::invalid("missing HTTP path"))?
    .to_owned();

  let mut content_length = 0usize;
  let mut authorization = None;
  loop {
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let line = line.trim_end_matches(['\r', '\n']);
    if line.is_empty() {
      break;
    }
    let Some((name, value)) = line.split_once(':') else {
      continue;
    };
    let value = value.trim();
    match name.trim().to_ascii_lowercase().as_str() {
      "content-length" => {
        content_length = value
          .parse::<usize>()
          .map_err(|_| CliFailure::invalid("Content-Length must be an unsigned integer"))?;
      }
      "authorization" => authorization = Some(value.to_owned()),
      "x-netron-token" => authorization = Some(format!("Bearer {value}")),
      _ => {}
    }
  }

  let mut body = vec![0u8; content_length];
  reader.read_exact(&mut body)?;
  let body = String::from_utf8(body)
    .map_err(|error| CliFailure::invalid(format!("HTTP body must be UTF-8: {error}")))?;
  Ok(HttpRequest {
    method,
    path,
    authorization,
    body,
  })
}

fn write_http_json(stream: &mut TcpStream, status: u16, value: &Value) -> Result<(), CliFailure> {
  let body = serde_json::to_string(value).map_err(CliFailure::from)?;
  write_http_response(stream, status, "application/json", body.as_bytes())
}

fn write_http_empty(stream: &mut TcpStream, status: u16) -> Result<(), CliFailure> {
  write_http_response(stream, status, "text/plain", b"")
}

fn write_http_response(
  stream: &mut TcpStream,
  status: u16,
  content_type: &str,
  body: &[u8],
) -> Result<(), CliFailure> {
  let reason = match status {
    200 => "OK",
    204 => "No Content",
    400 => "Bad Request",
    401 => "Unauthorized",
    405 => "Method Not Allowed",
    _ => "Error",
  };
  write!(
    stream,
    "HTTP/1.1 {status} {reason}\r\n\
     Content-Type: {content_type}\r\n\
     Content-Length: {}\r\n\
     Access-Control-Allow-Origin: *\r\n\
     Access-Control-Allow-Headers: content-type, authorization, x-netron-token\r\n\
     Access-Control-Allow-Methods: POST, OPTIONS, GET\r\n\
     Connection: close\r\n\
     \r\n",
    body.len()
  )?;
  stream.write_all(body)?;
  stream.flush()?;
  Ok(())
}

fn http_token() -> String {
  let mut bytes = [0u8; 16];
  if File::open("/dev/urandom")
    .and_then(|mut file| file.read_exact(&mut bytes))
    .is_ok()
  {
    return bytes_to_hex(&bytes);
  }
  let nanos = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .map_or(0, |duration| duration.as_nanos());
  format!("{:08x}{nanos:032x}", std::process::id())
}

fn bytes_to_hex(bytes: &[u8]) -> String {
  let mut hex = String::with_capacity(bytes.len() * 2);
  for byte in bytes {
    hex.push_str(&format!("{byte:02x}"));
  }
  hex
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

fn request_cancellation_key(request: &ServiceRequest) -> Result<Option<String>, CliFailure> {
  if let Some(value) = request.params.get("cancel_token") {
    return value_to_cancel_token(value).map(Some);
  }
  Ok(request.id.as_ref().and_then(request_id_cancel_token))
}

fn cancel_token_param(params: &Value) -> Result<String, CliFailure> {
  let value = params
    .get("cancel_token")
    .ok_or_else(|| CliFailure::invalid("cancel_token is required"))?;
  value_to_cancel_token(value)
}

fn value_to_cancel_token(value: &Value) -> Result<String, CliFailure> {
  value
    .as_str()
    .filter(|token| !token.is_empty())
    .map(str::to_owned)
    .ok_or_else(|| CliFailure::invalid("cancel_token must be a non-empty string"))
}

fn request_id_cancel_token(id: &Value) -> Option<String> {
  match id {
    Value::String(value) if !value.is_empty() => Some(value.clone()),
    Value::Number(value) => Some(value.to_string()),
    Value::Bool(value) => Some(value.to_string()),
    _ => None,
  }
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

fn optional_cursor_param(params: &Value) -> Result<Option<usize>, CliFailure> {
  match params.get("cursor") {
    None | Some(Value::Null) => Ok(None),
    Some(value) => value_to_usize(value, "cursor").map(Some),
  }
}

fn collapse_param(params: &Value) -> Result<CollapseMode, CliFailure> {
  match params.get("collapse") {
    None | Some(Value::Null) => Ok(CollapseMode::None),
    Some(Value::String(value)) => CollapseMode::parse(value)
      .ok_or_else(|| CliFailure::invalid("collapse must be 'none' or 'structural'")),
    Some(_) => Err(CliFailure::invalid("collapse must be a string")),
  }
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
    "onnx_repeated_block" => Ok(EntityHandle::OnnxRepeatedBlock {
      graph: usize_param(handle, "graph")?,
      group: usize_param(handle, "group")?,
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

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn slice_requests_register_cancellation_tokens() {
    let state = ServiceState::default();
    let request = ServiceRequest {
      id: Some(json!("slice-token")),
      method: "slice".to_owned(),
      params: json!({}),
    };

    let registration = state
      .register_request_cancellation(&request)
      .expect("registration succeeds")
      .expect("slice is cancelable");
    assert_eq!(registration.0, "slice-token");
    assert!(
      state
        .lock_cancellations()
        .expect("cancellation map")
        .contains_key("slice-token")
    );

    let canceled = state
      .cancel(&json!({ "cancel_token": "slice-token" }))
      .expect("cancel response");
    assert_eq!(canceled["canceled"], true);
    state
      .unregister_cancellation("slice-token")
      .expect("unregister token");
  }

  #[test]
  fn non_projection_requests_do_not_register_cancellation_tokens() {
    let state = ServiceState::default();
    let request = ServiceRequest {
      id: Some(json!("summary-token")),
      method: "summary".to_owned(),
      params: json!({}),
    };

    let registration = state
      .register_request_cancellation(&request)
      .expect("registration check succeeds");

    assert!(registration.is_none());
  }
}
