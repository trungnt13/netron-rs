use std::collections::HashMap;
use std::sync::Arc;

use crate::StringId;

#[derive(Debug, Default, Clone)]
pub struct StringInterner {
    ids: HashMap<Arc<str>, StringId>,
    strings: Vec<Arc<str>>,
}

impl StringInterner {
    pub fn intern(&mut self, value: impl AsRef<str>) -> StringId {
        let value = value.as_ref();
        if let Some(id) = self.ids.get(value) {
            return *id;
        }

        let id = StringId::new(self.strings.len());
        let stored = Arc::<str>::from(value);
        self.strings.push(stored.clone());
        self.ids.insert(stored, id);
        id
    }

    pub fn get(&self, id: StringId) -> &str {
        &self.strings[id.index()]
    }

    pub fn len(&self) -> usize {
        self.strings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }
}
