use std::io::Read;

use flate2::read::DeflateDecoder;
use netron_rs_core::ModelError;

pub(crate) struct ZipArchive<'a> {
    pub(crate) entries: Vec<ZipEntry<'a>>,
}

impl<'a> ZipArchive<'a> {
    pub(crate) fn open(data: &'a [u8]) -> Result<Self, ModelError> {
        if data.get(0..4) != Some(b"PK\x03\x04") {
            return Err(invalid("not a zip archive"));
        }
        Self::open_central(data).or_else(|_| Self::open_local(data))
    }

    fn open_local(data: &'a [u8]) -> Result<Self, ModelError> {
        let mut offset = 0;
        let mut entries = Vec::new();
        while data.get(offset..offset + 4) == Some(b"PK\x03\x04") {
            let header = data
                .get(offset..offset + 30)
                .ok_or_else(|| invalid("zip local header is truncated"))?;
            let flags = le_u16(header, 6);
            if flags & 0x08 != 0 {
                return Err(invalid("zip data descriptors are not supported"));
            }
            let method = le_u16(header, 8);
            let compressed_size = le_u32(header, 18) as usize;
            let uncompressed_size = le_u32(header, 22) as usize;
            let name_len = le_u16(header, 26) as usize;
            let extra_len = le_u16(header, 28) as usize;
            let name_start = offset + 30;
            let name_end = name_start
                .checked_add(name_len)
                .ok_or_else(|| invalid("zip file name offset overflows usize"))?;
            let extra_end = name_end
                .checked_add(extra_len)
                .ok_or_else(|| invalid("zip extra field offset overflows usize"))?;
            let data_end = extra_end
                .checked_add(compressed_size)
                .ok_or_else(|| invalid("zip data offset overflows usize"))?;
            let name = std::str::from_utf8(
                data.get(name_start..name_end)
                    .ok_or_else(|| invalid("zip file name is truncated"))?,
            )
            .map_err(|error| invalid(format!("zip file name: {error}")))?;
            let compressed = data
                .get(extra_end..data_end)
                .ok_or_else(|| invalid("zip entry data is truncated"))?;
            entries.push(ZipEntry {
                name,
                method,
                compressed,
                uncompressed_size,
            });
            offset = data_end;
        }
        Ok(Self { entries })
    }

    fn open_central(data: &'a [u8]) -> Result<Self, ModelError> {
        let eocd = data
            .windows(4)
            .rposition(|window| window == b"PK\x05\x06")
            .ok_or_else(|| invalid("zip central directory is missing"))?;
        let eocd_header = data
            .get(eocd..eocd + 22)
            .ok_or_else(|| invalid("zip end of central directory is truncated"))?;
        let entry_count = le_u16(eocd_header, 10) as usize;
        let central_offset = le_u32(eocd_header, 16) as usize;
        let mut offset = central_offset;
        let mut entries = Vec::new();

        for _ in 0..entry_count {
            let header = data
                .get(offset..offset + 46)
                .ok_or_else(|| invalid("zip central directory header is truncated"))?;
            if header.get(0..4) != Some(b"PK\x01\x02") {
                return Err(invalid("zip central directory header signature is invalid"));
            }
            let method = le_u16(header, 10);
            let mut compressed_size = le_u32(header, 20) as u64;
            let mut uncompressed_size = le_u32(header, 24) as u64;
            let name_len = le_u16(header, 28) as usize;
            let extra_len = le_u16(header, 30) as usize;
            let comment_len = le_u16(header, 32) as usize;
            let mut local_offset = le_u32(header, 42) as u64;
            let name_start = offset + 46;
            let name_end = name_start
                .checked_add(name_len)
                .ok_or_else(|| invalid("zip central file name offset overflows usize"))?;
            let next = name_end
                .checked_add(extra_len)
                .and_then(|value| value.checked_add(comment_len))
                .ok_or_else(|| invalid("zip central directory offset overflows usize"))?;
            let name = std::str::from_utf8(
                data.get(name_start..name_end)
                    .ok_or_else(|| invalid("zip central file name is truncated"))?,
            )
            .map_err(|error| invalid(format!("zip file name: {error}")))?;
            if compressed_size == u32::MAX as u64
                || uncompressed_size == u32::MAX as u64
                || local_offset == u32::MAX as u64
            {
                let extra = data
                    .get(name_end..name_end + extra_len)
                    .ok_or_else(|| invalid("zip central extra field is truncated"))?;
                let zip64 = zip64_extra_values(
                    extra,
                    uncompressed_size == u32::MAX as u64,
                    compressed_size == u32::MAX as u64,
                    local_offset == u32::MAX as u64,
                )?;
                if let Some(value) = zip64.uncompressed_size {
                    uncompressed_size = value;
                }
                if let Some(value) = zip64.compressed_size {
                    compressed_size = value;
                }
                if let Some(value) = zip64.local_offset {
                    local_offset = value;
                }
            }
            let compressed_size = usize::try_from(compressed_size)
                .map_err(|_| invalid("zip compressed size overflows usize"))?;
            let uncompressed_size = usize::try_from(uncompressed_size)
                .map_err(|_| invalid("zip uncompressed size overflows usize"))?;
            let local_offset = usize::try_from(local_offset)
                .map_err(|_| invalid("zip local offset overflows usize"))?;

            let local_header = data
                .get(local_offset..local_offset + 30)
                .ok_or_else(|| invalid("zip local header is truncated"))?;
            if local_header.get(0..4) != Some(b"PK\x03\x04") {
                return Err(invalid("zip local header signature is invalid"));
            }
            let local_name_len = le_u16(local_header, 26) as usize;
            let local_extra_len = le_u16(local_header, 28) as usize;
            let data_start = local_offset
                .checked_add(30)
                .and_then(|value| value.checked_add(local_name_len))
                .and_then(|value| value.checked_add(local_extra_len))
                .ok_or_else(|| invalid("zip local data offset overflows usize"))?;
            let data_end = data_start
                .checked_add(compressed_size)
                .ok_or_else(|| invalid("zip local data end overflows usize"))?;
            let compressed = data
                .get(data_start..data_end)
                .ok_or_else(|| invalid("zip entry data is truncated"))?;
            entries.push(ZipEntry {
                name,
                method,
                compressed,
                uncompressed_size,
            });
            offset = next;
        }

        Ok(Self { entries })
    }
}

pub(crate) struct ZipEntry<'a> {
    pub(crate) name: &'a str,
    method: u16,
    compressed: &'a [u8],
    uncompressed_size: usize,
}

impl ZipEntry<'_> {
    pub(crate) fn bytes(&self) -> Result<Vec<u8>, ModelError> {
        let bytes = match self.method {
            0 => self.compressed.to_vec(),
            8 => {
                let mut decoder = DeflateDecoder::new(self.compressed);
                let mut output = Vec::with_capacity(self.uncompressed_size);
                decoder
                    .read_to_end(&mut output)
                    .map_err(|error| invalid(format!("zip deflate decode failed: {error}")))?;
                output
            }
            method => return Err(invalid(format!("unsupported zip compression '{method}'"))),
        };
        if bytes.len() != self.uncompressed_size {
            return Err(invalid(format!(
                "zip entry '{}' decoded to {} bytes, expected {}",
                self.name,
                bytes.len(),
                self.uncompressed_size
            )));
        }
        Ok(bytes)
    }
}

struct Zip64ExtraValues {
    uncompressed_size: Option<u64>,
    compressed_size: Option<u64>,
    local_offset: Option<u64>,
}

fn zip64_extra_values(
    extra: &[u8],
    need_uncompressed_size: bool,
    need_compressed_size: bool,
    need_local_offset: bool,
) -> Result<Zip64ExtraValues, ModelError> {
    let mut offset = 0usize;
    while offset + 4 <= extra.len() {
        let header_id = le_u16(extra, offset);
        let len = le_u16(extra, offset + 2) as usize;
        let start = offset + 4;
        let end = start
            .checked_add(len)
            .ok_or_else(|| invalid("zip extra field length overflows usize"))?;
        let Some(payload) = extra.get(start..end) else {
            return Err(invalid("zip extra field is truncated"));
        };
        if header_id == 0x0001 {
            let mut payload_offset = 0usize;
            let uncompressed_size = if need_uncompressed_size {
                Some(read_zip64_field(payload, &mut payload_offset)?)
            } else {
                None
            };
            let compressed_size = if need_compressed_size {
                Some(read_zip64_field(payload, &mut payload_offset)?)
            } else {
                None
            };
            let local_offset = if need_local_offset {
                Some(read_zip64_field(payload, &mut payload_offset)?)
            } else {
                None
            };
            return Ok(Zip64ExtraValues {
                uncompressed_size,
                compressed_size,
                local_offset,
            });
        }
        offset = end;
    }
    Err(invalid("zip64 extra field is missing"))
}

fn read_zip64_field(payload: &[u8], offset: &mut usize) -> Result<u64, ModelError> {
    let value_offset = *offset;
    let end = value_offset
        .checked_add(8)
        .ok_or_else(|| invalid("zip64 field offset overflows usize"))?;
    if end > payload.len() {
        return Err(invalid("zip64 extra field is truncated"));
    }
    *offset = end;
    Ok(le_u64(payload, value_offset))
}

fn le_u16(data: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([data[offset], data[offset + 1]])
}

fn le_u32(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

fn le_u64(data: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
        data[offset + 4],
        data[offset + 5],
        data[offset + 6],
        data[offset + 7],
    ])
}

fn invalid(message: impl Into<String>) -> ModelError {
    ModelError::InvalidData {
        format: "ZIP",
        message: message.into(),
    }
}
