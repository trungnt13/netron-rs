macro_rules! id_type {
  ($name:ident) => {
    #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
    pub struct $name(u32);

    impl $name {
      pub fn new(index: usize) -> Self {
        assert!(index <= u32::MAX as usize);
        Self(index as u32)
      }

      pub fn index(self) -> usize {
        self.0 as usize
      }
    }
  };
}

id_type!(GraphId);
id_type!(NodeId);
id_type!(StringId);
id_type!(TensorId);
id_type!(ValueId);
