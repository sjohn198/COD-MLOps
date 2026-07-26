use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Default, Debug)]
#[serde(transparent)]
pub struct NetworkMap {
    pub layers: Vec<(i32, i32)>
}