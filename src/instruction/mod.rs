use std::sync::LazyLock;

use crate::cpu::Cpu;

mod format;
mod rv32i;

const OPCODE_SIZE: usize = 1 << 5;
type InsImpl = fn(u32, &mut Cpu);

#[derive(Clone, Copy)]
pub struct Instruction(pub u32);
impl Instruction {
    pub fn opcode(self) -> usize {
        (self.0 >> 2 & low_bitmask_u32(5)) as usize
    }
}

pub fn instruction_array() -> &'static [InsImpl; OPCODE_SIZE] {
    static GATHERED: LazyLock<[InsImpl; OPCODE_SIZE]> = LazyLock::new(|| {
        let mut arr = [nop as fn(u32, &mut Cpu); OPCODE_SIZE];
        for (opcode, ins_impl) in rv32i::OPCODE2INS_IMPL.iter() {
            arr[*opcode as usize] = *ins_impl;
        }
        arr
    });
    &*GATHERED
}

pub fn nop(_: u32, _: &mut Cpu) {

}

fn low_bitmask_u8(cnt: usize) -> u8 {
    assert!(cnt < 8);
    (1 << cnt) - 1
}

fn low_bitmask_u32(cnt: usize) -> u32 {
    assert!(cnt < 32);
    (1 << cnt) - 1
}