use color_eyre::Result;

use crate::instruction::{instruction_array, Instruction};

pub const N_CSR: usize = 4096;
pub const INST_SIZE: u32 = 4;

pub struct Cpu {
    pub reg: [u32; 32],
    pub csr: [u32; N_CSR],
    pub pc: u32,
    pub bus: Bus,
}

impl Default for Cpu {
    fn default() -> Self {
        Self {
            reg: Default::default(),
            csr: [0u32; N_CSR],
            pc: Default::default(),
            bus: Default::default(),
        }
    }
}

impl Cpu {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn execute(&mut self, ins_buf: &[u32]) -> Result<()> {
        let ins_arr = instruction_array();
        for ins in ins_buf {
            let ins = Instruction(*ins);
            let opcode = ins.opcode();
            ins_arr[opcode](ins.0, self);
            self.reg[0] = 0;
        }
        Ok(())
    }

    pub fn reset(&mut self) {
        std::mem::take(self);
    }
}

pub enum InstExecResult {
    Ok,
    Exception(InstException),
    Return(ReturnTarget),
    Pause,
    BadInst(String),
}

pub enum ReturnTarget {
    User,
    Machine,
}

pub enum InstException {
    EnvCall,
    Breakpoint,
}

pub struct Bus {
    base: u32,
    mem: Mem,
}

impl Bus {
    pub fn read(&self, addr: u32, buf: &mut [u8]) {
        let offset = (addr - self.base) as usize;
        let size = buf.len();
        buf.copy_from_slice(&self.mem.0[offset..offset + size]);
    }

    pub fn write(&mut self, addr: u32, data: &[u8]) {
        let offset = (addr - self.base) as usize;
        let size = data.len();
        let range = &mut self.mem.0[offset..offset + size];
        range.copy_from_slice(data);
    }
}

impl Default for Bus {
    fn default() -> Self {
        Self {
            base: 0x1000,
            mem: Mem(vec![0; 0x10000]),
        }
    }
}

pub struct Mem(pub Vec<u8>);
