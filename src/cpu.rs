use color_eyre::Result;

use crate::instruction::{instruction_array, Instruction};

#[derive(Default)]
pub struct Cpu {
    pub reg: [u64; 32],
    pub pc: u64,
    pub bus: Bus,
}

pub struct Bus {
    mem: Mem,
}

impl Bus {
    pub fn read(&self, addr: u64, buf: &mut [u8]) {
        let addr = addr as usize;
        let size = buf.len();
        buf.copy_from_slice(&self.mem.0[addr..addr+size]);
    }

    pub fn write(&mut self, addr: u64, data: &[u8]) {
        let addr = addr as usize;
        let size = data.len();
        let range = &mut self.mem.0[addr..addr+size];
        range.copy_from_slice(data);
    }
}

impl Default for Bus {
    fn default() -> Self {
        Self { mem: Mem(vec![0; 1024]) }
    }
}

pub struct Mem(pub Vec<u8>);

impl Cpu {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn execute(&mut self, ins_buf: &[u32]) -> Result<()>{
        let ins_arr = instruction_array();
        for ins in ins_buf {
            let ins = Instruction(*ins);
            ins_arr[ins.opcode()](ins.0, self);
            self.reg[0] = 0;
        }
        Ok(())
    }
}