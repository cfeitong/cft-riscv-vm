use super::{
    format::{IFormat, RFormat, SBFormat, SFormat, UFormat, UJFormat},
    InsImpl,
};
use crate::cpu::{Cpu, InstExecResult, ReturnTarget, INST_SIZE};
use std::{collections::BTreeMap, sync::LazyLock};

const SHIFT_MASK: u32 = 0b11111;

pub static OPCODE2INS_IMPL: LazyLock<BTreeMap<u8, InsImpl>> = LazyLock::new(|| {
    let mut map = BTreeMap::new();
    register_instructions(&mut map);
    map
});

pub fn lui(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = UFormat(instruction);
    let imm = f.imm();
    cpu.reg[f.rd() as usize] = sext(imm << 12, 31);
    InstExecResult::Ok
}

pub fn auipc(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = UFormat(instruction);
    let imm = f.imm();
    cpu.reg[f.rd() as usize] = cpu.pc + sext(imm << 12, 31);
    InstExecResult::Ok
}

pub fn addi(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = IFormat(instruction);
    let rd = f.rd() as usize;
    let rs1 = f.rs1() as usize;
    let imm = sext(f.imm(), 11);
    cpu.reg[rd] = cpu.reg[rs1].wrapping_add(imm);
    InstExecResult::Ok
}

pub fn slti(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = IFormat(instruction);
    let rd = f.rd() as usize;
    let rs1 = f.rs1() as usize;
    let imm = sext(f.imm(), 11) as i32;
    let rs1_v = cpu.reg[rs1] as i32;
    cpu.reg[rd] = if rs1_v < imm { 1 } else { 0 };
    InstExecResult::Ok
}

pub fn sltiu(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = IFormat(instruction);
    let rd = f.rd() as usize;
    let rs1 = f.rs1() as usize;
    let imm = f.imm();
    cpu.reg[rd] = if cpu.reg[rs1] < imm { 1 } else { 0 };
    InstExecResult::Ok
}

pub fn xori(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = IFormat(instruction);
    let rd = f.rd() as usize;
    let rs1 = f.rs1() as usize;
    let imm = sext(f.imm(), 11);
    cpu.reg[rd] = cpu.reg[rs1] ^ imm;
    InstExecResult::Ok
}

pub fn ori(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = IFormat(instruction);
    let rd = f.rd() as usize;
    let rs1 = f.rs1() as usize;
    let imm = sext(f.imm(), 11);
    cpu.reg[rd] = cpu.reg[rs1] | imm;
    InstExecResult::Ok
}

pub fn andi(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = IFormat(instruction);
    let rd = f.rd() as usize;
    let rs1 = f.rs1() as usize;
    let imm = sext(f.imm(), 11);
    cpu.reg[rd] = cpu.reg[rs1] & imm;
    InstExecResult::Ok
}

pub fn slli(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = IFormat(instruction);
    let rd = f.rd() as usize;
    let rs1 = f.rs1() as usize;
    let shamt = f.imm() & SHIFT_MASK;
    cpu.reg[rd] = cpu.reg[rs1] << shamt;
    InstExecResult::Ok
}

pub fn srli(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = IFormat(instruction);
    let rd = f.rd() as usize;
    let rs1 = f.rs1() as usize;
    let shamt = f.imm() & SHIFT_MASK;
    cpu.reg[rd] = cpu.reg[rs1] >> shamt;
    InstExecResult::Ok
}

pub fn srai(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = IFormat(instruction);
    let rd = f.rd() as usize;
    let rs1 = f.rs1() as usize;
    let shamt = f.imm() & SHIFT_MASK;
    cpu.reg[rd] = (cpu.reg[rs1] as i32 >> shamt) as u32;
    InstExecResult::Ok
}

pub fn add(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = RFormat(instruction);
    let rd = f.rd() as usize;
    let rs1 = f.rs1() as usize;
    let rs2 = f.rs2() as usize;
    cpu.reg[rd] = cpu.reg[rs1] + cpu.reg[rs2];
    InstExecResult::Ok
}

pub fn sub(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = RFormat(instruction);
    let rd = f.rd() as usize;
    let rs1 = f.rs1() as usize;
    let rs2 = f.rs2() as usize;
    cpu.reg[rd] = cpu.reg[rs1] - cpu.reg[rs2];
    InstExecResult::Ok
}

pub fn sll(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = RFormat(instruction);
    let rd = f.rd() as usize;
    let rs1 = f.rs1() as usize;
    let rs2 = f.rs2() as usize;
    cpu.reg[rd] = cpu.reg[rs1] << (cpu.reg[rs2] & SHIFT_MASK);
    InstExecResult::Ok
}

pub fn slt(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = RFormat(instruction);
    let rd = f.rd() as usize;
    let rs1 = f.rs1() as usize;
    let rs1_v = cpu.reg[rs1] as isize;
    let rs2 = f.rs2() as usize;
    let rs2_v = cpu.reg[rs2] as isize;
    cpu.reg[rd] = if rs1_v < rs2_v { 1 } else { 0 };
    InstExecResult::Ok
}

pub fn sltu(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = RFormat(instruction);
    let rd = f.rd() as usize;
    let rs1 = f.rs1() as usize;
    let rs1_v = cpu.reg[rs1] as usize;
    let rs2 = f.rs2() as usize;
    let rs2_v = cpu.reg[rs2] as usize;
    cpu.reg[rd] = if rs1_v < rs2_v { 1 } else { 0 };
    InstExecResult::Ok
}

pub fn xor(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = RFormat(instruction);
    let rd = f.rd() as usize;
    let rs1 = f.rs1() as usize;
    let rs2 = f.rs2() as usize;
    cpu.reg[rd] = cpu.reg[rs1] ^ cpu.reg[rs2];
    InstExecResult::Ok
}

pub fn srl(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = RFormat(instruction);
    let rd = f.rd() as usize;
    let rs1 = f.rs1() as usize;
    let rs2 = f.rs2() as usize;
    cpu.reg[rd] = cpu.reg[rs1] >> (cpu.reg[rs2] & SHIFT_MASK);
    InstExecResult::Ok
}

pub fn sra(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = RFormat(instruction);
    let rd = f.rd() as usize;
    let rs1 = f.rs1() as usize;
    let rs2 = f.rs2() as usize;
    cpu.reg[rd] = (cpu.reg[rs1] as i32 >> (cpu.reg[rs2] & SHIFT_MASK)) as u32;
    InstExecResult::Ok
}

pub fn or(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = RFormat(instruction);
    let rd = f.rd() as usize;
    let rs1 = f.rs1() as usize;
    let rs2 = f.rs2() as usize;
    cpu.reg[rd] = cpu.reg[rs1] | cpu.reg[rs2];
    InstExecResult::Ok
}

pub fn and(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = RFormat(instruction);
    let rd = f.rd() as usize;
    let rs1 = f.rs1() as usize;
    let rs2 = f.rs2() as usize;
    cpu.reg[rd] = cpu.reg[rs1] & cpu.reg[rs2];
    InstExecResult::Ok
}

pub fn fence(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    InstExecResult::Ok
}

pub fn fence_i(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    InstExecResult::Ok
}

pub fn csrrw(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = IFormat(instruction);
    let rd = f.rd() as usize;
    let rs1 = f.rs1() as usize;
    let csr = f.imm() as usize;
    cpu.reg[rd] = cpu.csr[csr];
    cpu.csr[csr] = cpu.reg[rs1];
    InstExecResult::Ok
}

pub fn csrrs(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = IFormat(instruction);
    let rd = f.rd() as usize;
    let rs1 = f.rs1() as usize;
    let csr = f.imm() as usize;
    cpu.reg[rd] = cpu.csr[csr];
    cpu.csr[csr] = cpu.reg[rs1] | cpu.reg[rd];
    InstExecResult::Ok
}

pub fn csrrc(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = IFormat(instruction);
    let rd = f.rd() as usize;
    let rs1 = f.rs1() as usize;
    let csr = f.imm() as usize;
    cpu.reg[rd] = cpu.csr[csr];
    cpu.csr[csr] = !cpu.reg[rs1] & cpu.reg[rd];
    InstExecResult::Ok
}

pub fn csrrwi(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = IFormat(instruction);
    let rd = f.rd() as usize;
    let rs1 = f.rs1() as usize;
    let zimm = rs1 as u32;
    let csr = f.imm() as usize;
    cpu.reg[rd] = cpu.csr[csr];
    cpu.csr[csr] = zimm;
    InstExecResult::Ok
}

pub fn csrrsi(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = IFormat(instruction);
    let rd = f.rd() as usize;
    let rs1 = f.rs1() as usize;
    let zimm = rs1 as u32;
    let csr = f.imm() as usize;
    cpu.reg[rd] = cpu.csr[csr];
    cpu.csr[csr] = zimm | cpu.reg[rd];
    InstExecResult::Ok
}

pub fn csrrci(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = IFormat(instruction);
    let rd = f.rd() as usize;
    let rs1 = f.rs1() as usize;
    let zimm = rs1 as u32;
    let csr = f.imm() as usize;
    cpu.reg[rd] = cpu.csr[csr];
    cpu.csr[csr] = !zimm & cpu.reg[rd];
    InstExecResult::Ok
}

pub fn ecall(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    InstExecResult::Exception(crate::cpu::InstException::EnvCall)
}

pub fn ebreak(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    InstExecResult::Exception(crate::cpu::InstException::Breakpoint)
}

pub fn uret(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    InstExecResult::Return(ReturnTarget::User)
}

pub fn sret(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    InstExecResult::Return(ReturnTarget::User)
}

pub fn mret(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    InstExecResult::Return(ReturnTarget::Machine)
}

pub fn wfi(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    InstExecResult::Pause
}

pub fn sfence_vma(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    InstExecResult::Ok
}

pub fn l(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = IFormat(instruction);
    match f.funct3() {
        0b000 => load_mem::<1>(f, cpu, sext), // lb
        0b001 => load_mem::<2>(f, cpu, sext), // lh
        0b010 => load_mem::<4>(f, cpu, sext), // lw
        0b100 => load_mem::<1>(f, cpu, uext), // lbu
        0b101 => load_mem::<2>(f, cpu, uext), // lhu
        _ => {
            return InstExecResult::BadInst(format!(
                "invalid load instrunction func4 0b{:b}",
                f.funct3()
            ));
        }
    }
    InstExecResult::Ok
}

pub fn s(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = SFormat(instruction);
    match f.funct3() {
        0b000 => store_mem::<1>(f, cpu), // sb
        0b001 => store_mem::<2>(f, cpu), // sh
        0b010 => store_mem::<4>(f, cpu), // sw
        _ => {
            return InstExecResult::BadInst(format!(
                "invalid store instrunction func4 0b{:b}",
                f.funct3()
            ));
        }
    }
    InstExecResult::Ok
}

pub fn jal(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = UJFormat(instruction);
    let offset = sext(f.imm(), 20);
    let rd = f.rd() as usize;
    cpu.reg[rd] = cpu.pc + INST_SIZE;
    cpu.pc = cpu.pc.wrapping_add(offset);
    InstExecResult::Ok
}

pub fn jalr(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = IFormat(instruction);
    let offset = sext(f.imm(), 11);
    let rd = f.rd() as usize;
    let rs1 = f.rs1() as usize;
    cpu.reg[rd] = cpu.pc + INST_SIZE;
    cpu.pc = (cpu.reg[rs1].wrapping_add(offset)) & !0b1;
    InstExecResult::Ok
}

pub fn beq(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = SBFormat(instruction);
    let offset = sext(f.imm(), 12);
    let rs1 = f.rs1() as usize;
    let rs2 = f.rs2() as usize;
    if cpu.reg[rs1] == cpu.reg[rs2] {
        cpu.pc = cpu.pc.wrapping_add(offset)
    }
    InstExecResult::Ok
}

pub fn bne(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = SBFormat(instruction);
    let offset = sext(f.imm(), 12);
    let rs1 = f.rs1() as usize;
    let rs2 = f.rs2() as usize;
    if cpu.reg[rs1] != cpu.reg[rs2] {
        cpu.pc = cpu.pc.wrapping_add(offset);
    }
    InstExecResult::Ok
}

pub fn blt(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = SBFormat(instruction);
    let offset = sext(f.imm(), 12);
    let rs1 = f.rs1() as usize;
    let rs2 = f.rs2() as usize;
    if (cpu.reg[rs1] as i32) < (cpu.reg[rs2] as i32) {
        cpu.pc = cpu.pc.wrapping_add(offset);
    }
    InstExecResult::Ok
}

pub fn bge(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = SBFormat(instruction);
    let offset = sext(f.imm(), 12);
    let rs1 = f.rs1() as usize;
    let rs2 = f.rs2() as usize;
    if (cpu.reg[rs1] as i32) >= (cpu.reg[rs2] as i32) {
        cpu.pc = cpu.pc.wrapping_add(offset);
    }
    InstExecResult::Ok
}

pub fn bltu(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = SBFormat(instruction);
    let offset = sext(f.imm(), 12);
    let rs1 = f.rs1() as usize;
    let rs2 = f.rs2() as usize;
    if cpu.reg[rs1] < cpu.reg[rs2] {
        cpu.pc = cpu.pc.wrapping_add(offset);
    }
    InstExecResult::Ok
}

pub fn bgeu(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = SBFormat(instruction);
    let offset = sext(f.imm(), 12);
    let rs1 = f.rs1() as usize;
    let rs2 = f.rs2() as usize;
    if cpu.reg[rs1] >= cpu.reg[rs2] {
        cpu.pc = cpu.pc.wrapping_add(offset);
    }
    InstExecResult::Ok
}

fn load_mem<const BYTES: usize>(
    instruction: IFormat,
    cpu: &mut Cpu,
    ext_fn: impl Fn(u32, usize) -> u32,
) {
    let f = instruction;
    let offset = sext(f.imm(), 11);
    let rs1 = f.rs1() as usize;
    let rd = f.rd() as usize;
    let addr = cpu.reg[rs1].wrapping_add(offset);
    let mut buf = [0; std::mem::size_of::<u32>()];
    cpu.bus.read(addr, &mut buf[0..BYTES]);
    let value = u32::from_le_bytes(buf);
    let msb = (8 << BYTES) - 1;
    cpu.reg[rd] = ext_fn(value, msb);
}

fn store_mem<const BYTES: usize>(instruction: SFormat, cpu: &mut Cpu) {
    let f = instruction;
    let offset = sext(f.imm(), 11);
    let rs1 = f.rs1() as usize;
    let rs2 = f.rs2() as usize;
    let addr = cpu.reg[rs1].wrapping_add(offset);
    let buf = cpu.reg[rs2].to_le_bytes();
    cpu.bus.write(addr, &buf[0..BYTES]);
}

fn bit_op(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = IFormat(instruction);
    const ARITHMATIC_MASK: u32 = 1 << 11;
    match f.funct3() {
        0b000 => addi(instruction, cpu),
        0b010 => slti(instruction, cpu),
        0b011 => sltiu(instruction, cpu),
        0b100 => xori(instruction, cpu),
        0b110 => ori(instruction, cpu),
        0b111 => andi(instruction, cpu),
        0b101 => {
            if f.imm() & ARITHMATIC_MASK == 0 {
                srli(instruction, cpu)
            } else {
                srai(instruction, cpu)
            }
        }
        funct3 @ _ => InstExecResult::BadInst(format!(
            "invalid bit operation instrunction funct3 0b{funct3:b}"
        )),
    }
}

fn alu_op(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = IFormat(instruction);
    const SUB_MASK: u32 = 1 << 11;
    const ARITHMATIC_MASK: u32 = 1 << 11;
    match f.funct3() {
        0b000 => {
            if f.imm() & SUB_MASK == 0 {
                add(instruction, cpu)
            } else {
                sub(instruction, cpu)
            }
        }
        0b001 => sll(instruction, cpu),
        0b010 => slt(instruction, cpu),
        0b011 => sltu(instruction, cpu),
        0b100 => xor(instruction, cpu),
        0b101 => {
            if f.imm() & ARITHMATIC_MASK != 0 {
                srl(instruction, cpu)
            } else {
                sra(instruction, cpu)
            }
        }
        0b110 => or(instruction, cpu),
        0b111 => and(instruction, cpu),
        funct3 @ _ => InstExecResult::BadInst(format!(
            "invalid arithmatic operation instrunction funct3 0b{funct3:b}"
        )),
    }
}

fn fence_op(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = IFormat(instruction);
    match f.funct3() {
        0b000 => fence(instruction, cpu),
        0b001 => fence_i(instruction, cpu),
        funct3 @ _ => InstExecResult::BadInst(format!(
            "invalid fence operation instrunction funct3 0b{funct3:b}"
        )),
    }
}

fn special_op(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = IFormat(instruction);
    match f.funct3() {
        0b000 => env_op(instruction, cpu),
        0b001 => csrrw(instruction, cpu),
        0b010 => csrrs(instruction, cpu),
        0b011 => csrrc(instruction, cpu),
        0b101 => csrrwi(instruction, cpu),
        0b110 => csrrsi(instruction, cpu),
        0b111 => csrrci(instruction, cpu),
        funct3 @ _ => InstExecResult::BadInst(format!(
            "invalid csr operation instrunction funct3 0b{funct3:b}"
        )),
    }
}

fn env_op(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    match instruction {
        0b00000_00_00000_00000_000_00000_11100_11 => ecall(instruction, cpu),
        0b00000_00_00001_00000_000_00000_11100_11 => ebreak(instruction, cpu),
        0b00000_00_00010_00000_000_00000_11100_11 => uret(instruction, cpu),
        0b00010_00_00010_00000_000_00000_11100_11 => sret(instruction, cpu),
        0b00110_00_00010_00000_000_00000_11100_11 => mret(instruction, cpu),
        0b00010_00_00101_00000_000_00000_11100_11 => wfi(instruction, cpu),
        inst if inst & 0b00010_01_00000_00000_000_00000_11100_11 != 0 => {
            sfence_vma(instruction, cpu)
        }
        0b00010_01_00101_00000_000_00000_11100_11 => wfi(instruction, cpu),
        _ => InstExecResult::BadInst(format!("invalid special instrunction 0b{instruction:031b}")),
    }
}

fn branch_op(instruction: u32, cpu: &mut Cpu) -> InstExecResult {
    let f = SBFormat(instruction);
    match f.funct3() {
        0b000 => beq(instruction, cpu),
        0b001 => bne(instruction, cpu),
        0b100 => blt(instruction, cpu),
        0b101 => bge(instruction, cpu),
        0b110 => bltu(instruction, cpu),
        0b111 => bgeu(instruction, cpu),
        funct3 @ _ => InstExecResult::BadInst(format!(
            "invalid branch operation instrunction funct3 0b{funct3:b}"
        )),
    }
}

fn register_instructions(map: &mut BTreeMap<u8, InsImpl>) {
    map.insert(0b01101, lui);
    map.insert(0b00101, auipc);
    map.insert(0b00100, bit_op);
    map.insert(0b01100, alu_op);
    map.insert(0b00011, fence_op);
    map.insert(0b11100, special_op);
    map.insert(0b00000, l);
    map.insert(0b01000, s);
    map.insert(0b11011, jal);
    map.insert(0b11001, jalr);
    map.insert(0b11000, branch_op);
}

fn sext(imm: u32, msb: usize) -> u32 {
    let m = 1u32.wrapping_shl(msb as u32);
    if imm & m > 0 {
        (!(m - 1 | m)) | imm
    } else {
        imm
    }
}

fn uext(imm: u32, _: usize) -> u32 {
    imm
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::process::Command;

    use color_eyre::eyre::ContextCompat;
    use color_eyre::Result;
    use object::{Object, ObjectSection};

    use super::*;

    fn call_external_command(cmd: &mut Command) -> color_eyre::Result<()> {
        let status = cmd.spawn()?.wait()?;
        if !status.success() {
            color_eyre::eyre::bail!("fail to call {:?}", cmd.get_program().to_str());
        }
        Ok(())
    }

    fn assemble(code: &str) -> Result<Vec<u32>> {
        let mut source = tempfile::Builder::new().suffix(".s").tempfile()?;
        let mut obj_path = source.path().to_owned();
        obj_path.set_extension("o");
        source.write_all(code.as_bytes())?;
        source.write_all(b"\n")?;
        call_external_command(
            Command::new("riscv64-linux-gnu-as")
                .args(["-march", "rv32i"])
                .arg(source.path())
                .arg("-o")
                .arg(&obj_path),
        )?;
        let obj_bin = std::fs::read(&obj_path)?;
        let obj_file = object::read::File::parse(&*obj_bin)?;
        let text_sec = obj_file
            .section_by_name(".text")
            .wrap_err("no .text section found")?;
        let data = text_sec.data()?;
        let ins_buf: Result<Vec<_>> = data
            .chunks_exact(4)
            .map(|b| Ok(u32::from_le_bytes(b.try_into()?)))
            .collect();
        Ok(ins_buf?)
    }

    #[test]
    fn test_lui() -> Result<()> {
        let mut cpu = Cpu::new();

        let code = assemble("lui x16, 2")?;
        cpu.execute(&code)?;
        assert_eq!(cpu.reg[16], 8192);

        let code = assemble("lui x17, 0x0FFFFF")?;
        cpu.execute(&code)?;
        assert_eq!(cpu.reg[17], !0x0FFF);
        Ok(())
    }

    #[test]
    fn test_auipc() -> Result<()> {
        let mut cpu = Cpu::new();
        let code = assemble("auipc x16, 2")?;
        cpu.execute(&code)?;
        assert_eq!(cpu.reg[16], 0x2000);
        Ok(())
    }

    #[test]
    fn test_addi() -> Result<()> {
        let mut cpu = Cpu::new();
        let code = assemble("addi x31, x0, 4")?;
        cpu.execute(&code)?;
        assert_eq!(cpu.reg[31], 4);
        Ok(())
    }

    #[test]
    fn test_slti() -> Result<()> {
        let mut cpu = Cpu::new();
        let code = assemble(
            r#"
            addi x16, x0, -5
            slti x17, x16, -2
            "#,
        )?;
        cpu.execute(&code)?;
        assert_eq!(cpu.reg[16], -5i32 as u32);
        assert_eq!(cpu.reg[17], 1);
        Ok(())
    }

    #[test]
    fn test_sltiu() -> Result<()> {
        let mut cpu = Cpu::new();
        let code = assemble(
            r#"
            addi x16, x0, 2
            sltiu x17, x16, 5
            "#,
        )?;
        cpu.execute(&code)?;
        assert_eq!(cpu.reg[16], 2);
        assert_eq!(cpu.reg[17], 1);
        Ok(())
    }

    #[test]
    fn test_xori() -> Result<()> {
        let mut cpu = Cpu::new();
        let code = assemble(
            r#"
            addi x16, x0, 3
            xori x17, x16, 6
            "#,
        )?;
        cpu.execute(&code)?;
        assert_eq!(cpu.reg[16], 3);
        assert_eq!(cpu.reg[17], 5);
        Ok(())
    }

    #[test]
    fn test_ori() -> Result<()> {
        let mut cpu = Cpu::new();
        let code = assemble(
            r#"
            addi x16, x0, 3
            ori x17, x16, 6
            "#,
        )?;
        cpu.execute(&code)?;
        assert_eq!(cpu.reg[16], 3);
        assert_eq!(cpu.reg[17], 7);
        Ok(())
    }

    #[test]
    fn test_l() -> Result<()> {
        tracing_subscriber::fmt::init();
        let mut cpu = Cpu::new();
        cpu.bus.write(4, &[0x12, 0x34, 0x56, 0x78]);
        let code = assemble(
            r#"
            addi x16, x0, 5
            addi x17, x0, 3
            lb x18, 4(x0)
            lh x19, 4(x0)
            lw x20, 4(x0)
        "#,
        )?;
        cpu.execute(&code)?;
        assert_eq!(cpu.reg[16], 5);
        assert_eq!(cpu.reg[17], 3);
        assert_eq!(cpu.reg[18], 0x12);
        assert_eq!(cpu.reg[19], 0x3412);
        assert_eq!(cpu.reg[20], 0x78563412);
        Ok(())
    }

    #[test]
    fn test_lb_rd_offset_rs1() -> Result<()> {
        let mut cpu = Cpu::new();
        // Set x16 to a valid memory address, e.g., 0x1000
        cpu.reg[16] = 0x1000;
        // Write data -109 (0x93) to memory address 0x1004
        cpu.bus.write(0x1004, &[-109i8 as u8]);
        let code = assemble(
            r#"
        addi x17, x16, 8
        lb x18, -4(x17)
        "#,
        )?;
        cpu.execute(&code)?;
        assert_eq!(cpu.reg[17], 0x1008);
        assert_eq!(cpu.reg[18], -109i8 as u32);
        Ok(())
    }

    #[test]
    fn test_lh_rd_offset_rs1() -> Result<()> {
        let mut cpu = Cpu::new();
        cpu.reg[16] = 0x1000;
        // Write data 0x0893 to memory address 0x1004
        cpu.bus.write(0x1004, &0x0893u16.to_le_bytes());
        let code = assemble(
            r#"
        addi x17, x0, 4
        lh x18, 0(x16)
        "#,
        )?;
        cpu.execute(&code)?;
        assert_eq!(cpu.reg[17], 4);
        assert_eq!(cpu.reg[18], 0x0893);
        Ok(())
    }

    #[test]
    fn test_lw_rd_offset_rs1() -> Result<()> {
        let mut cpu = Cpu::new();
        cpu.reg[16] = 0x1000;
        // Write data 0x300893 to memory address 0x1004
        cpu.bus.write(0x1004, &0x300893u32.to_le_bytes());
        let code = assemble(
            r#"
        addi x17, x0, 4
        lw x18, 0(x16)
        "#,
        )?;
        cpu.execute(&code)?;
        assert_eq!(cpu.reg[17], 4);
        assert_eq!(cpu.reg[18], 0x300893);
        Ok(())
    }

    #[test]
    fn test_lbu_rd_offset_rs1() -> Result<()> {
        let mut cpu = Cpu::new();
        cpu.reg[16] = 0x1000;
        // Write data 0x93 to memory address 0x1004
        cpu.bus.write(0x1004, &[0x93]);
        let code = assemble(
            r#"
        addi x17, x0, 4
        lbu x18, 0(x16)
        "#,
        )?;
        cpu.execute(&code)?;
        assert_eq!(cpu.reg[17], 4);
        assert_eq!(cpu.reg[18], 0x93);
        Ok(())
    }

    #[test]
    fn test_lhu_rd_offset_rs1() -> Result<()> {
        let mut cpu = Cpu::new();
        cpu.reg[16] = 0x1000;
        // Write data 0x0893 to memory address 0x1004
        cpu.bus.write(0x1004, &0x0893u16.to_le_bytes());
        let code = assemble(
            r#"
        addi x17, x0, 4
        lhu x18, 0(x16)
        "#,
        )?;
        cpu.execute(&code)?;
        assert_eq!(cpu.reg[17], 4);
        assert_eq!(cpu.reg[18], 0x0893);
        Ok(())
    }

    #[test]
    fn test_slli_rd_rs1_imm() -> Result<()> {
        let mut cpu = Cpu::new();
        let code = assemble(
            r#"
        addi x16, x0, 2
        slli x17, x16, 3
        "#,
        )?;
        cpu.execute(&code)?;
        assert_eq!(cpu.reg[16], 2);
        assert_eq!(cpu.reg[17], 16);
        Ok(())
    }

    #[test]
    fn test_srai_rd_rs1_imm() -> Result<()> {
        let mut cpu = Cpu::new();
        let code = assemble(
            r#"
        addi x16, x0, -8
        srai x17, x16, 2
        "#,
        )?;
        cpu.execute(&code)?;
        assert_eq!(cpu.reg[16], (-8i32) as u32);
        assert_eq!(cpu.reg[17], (-2i32) as u32);
        Ok(())
    }

    #[test]
    fn test_srli_rd_rs1_imm() -> Result<()> {
        let mut cpu = Cpu::new();
        let code = assemble(
            r#"
        addi x16, x0, 8
        srli x17, x16, 2
        "#,
        )?;
        cpu.execute(&code)?;
        assert_eq!(cpu.reg[16], 8);
        assert_eq!(cpu.reg[17], 2);
        Ok(())
    }

    #[test]
    fn test_andi_rd_rs1_imm() -> Result<()> {
        let mut cpu = Cpu::new();
        let code = assemble(
            r#"
        addi x16, x0, 4
        andi x17, x16, 7
        "#,
        )?;
        cpu.execute(&code)?;
        assert_eq!(cpu.reg[16], 4);
        assert_eq!(cpu.reg[17], 4);
        Ok(())
    }

    #[test]
    fn test_auipc_rd_imm() -> Result<()> {
        let mut cpu = Cpu::new();
        cpu.pc = 0x8000_0000;
        let code = assemble(
            r#"
        auipc x16, 2
        "#,
        )?;
        cpu.execute(&code)?;
        assert_eq!(cpu.reg[16], 0x8000_2000);
        Ok(())
    }

    #[test]
    fn test_add_rd_rs1_rs2() -> Result<()> {
        let mut cpu = Cpu::new();
        let code = assemble(
            r#"
        addi x3, x0, 5
        addi x4, x0, 6
        add x2, x3, x4
        "#,
        )?;
        cpu.execute(&code)?;
        assert_eq!(cpu.reg[2], 11);
        assert_eq!(cpu.reg[3], 5);
        assert_eq!(cpu.reg[4], 6);
        Ok(())
    }

    #[test]
    fn test_sub_rd_rs1_rs2() -> Result<()> {
        let mut cpu = Cpu::new();
        let code = assemble(
            r#"
        addi x3, x0, 5
        addi x4, x0, 6
        sub x2, x3, x4
        "#,
        )?;
        cpu.execute(&code)?;
        assert_eq!(cpu.reg[2], (-1i32) as u32);
        assert_eq!(cpu.reg[3], 5);
        assert_eq!(cpu.reg[4], 6);
        Ok(())
    }

    #[test]
    fn test_sll_rd_rs1_rs2() -> Result<()> {
        let mut cpu = Cpu::new();
        let code = assemble(
            r#"
        addi x16, x0, 8
        addi x17, x0, 2
        sll x18, x16, x17
        "#,
        )?;
        cpu.execute(&code)?;
        assert_eq!(cpu.reg[16], 8);
        assert_eq!(cpu.reg[17], 2);
        assert_eq!(cpu.reg[18], 32);
        Ok(())
    }

    #[test]
    fn test_slt_rd_rs1_rs2() -> Result<()> {
        let mut cpu = Cpu::new();
        let code = assemble(
            r#"
        addi x16, x0, -8
        addi x17, x0, 2
        slt x18, x16, x17
        "#,
        )?;
        cpu.execute(&code)?;
        assert_eq!(cpu.reg[16], (-8i32) as u32);
        assert_eq!(cpu.reg[17], 2);
        assert_eq!(cpu.reg[18], 1);
        Ok(())
    }

    #[test]
    fn test_sltu_rd_rs1_rs2() -> Result<()> {
        let mut cpu = Cpu::new();
        let code = assemble(
            r#"
        addi x16, x0, 8
        addi x17, x0, 2
        sltu x18, x17, x16
        "#,
        )?;
        cpu.execute(&code)?;
        assert_eq!(cpu.reg[16], 8);
        assert_eq!(cpu.reg[17], 2);
        assert_eq!(cpu.reg[18], 1);
        Ok(())
    }

    #[test]
    fn test_xor_rd_rs1_rs2() -> Result<()> {
        let mut cpu = Cpu::new();
        let code = assemble(
            r#"
        addi x16, x0, 3
        addi x17, x0, 6
        xor x18, x16, x17
        "#,
        )?;
        cpu.execute(&code)?;
        assert_eq!(cpu.reg[16], 3);
        assert_eq!(cpu.reg[17], 6);
        assert_eq!(cpu.reg[18], 5);
        Ok(())
    }

    #[test]
    fn test_srl_rd_rs1_rs2() -> Result<()> {
        let mut cpu = Cpu::new();
        let code = assemble(
            r#"
        addi x16, x0, 16
        addi x17, x0, 2
        srl x18, x16, x17
        "#,
        )?;
        cpu.execute(&code)?;
        assert_eq!(cpu.reg[16], 16);
        assert_eq!(cpu.reg[17], 2);
        assert_eq!(cpu.reg[18], 4);
        Ok(())
    }

    #[test]
    fn test_sra_rd_rs1_rs2() -> Result<()> {
        let mut cpu = Cpu::new();
        let code = assemble(
            r#"
        addi x16, x0, -16
        addi x17, x0, 2
        sra x18, x16, x17
        "#,
        )?;
        cpu.execute(&code)?;
        assert_eq!(cpu.reg[16], (-16i32) as u32);
        assert_eq!(cpu.reg[17], 2);
        assert_eq!(cpu.reg[18], (-4i32) as u32);
        Ok(())
    }

    #[test]
    fn test_or_rd_rs1_rs2() -> Result<()> {
        let mut cpu = Cpu::new();
        let code = assemble(
            r#"
        addi x16, x0, 3
        addi x17, x0, 5
        or x18, x16, x17
        "#,
        )?;
        cpu.execute(&code)?;
        assert_eq!(cpu.reg[16], 3);
        assert_eq!(cpu.reg[17], 5);
        assert_eq!(cpu.reg[18], 7);
        Ok(())
    }

    #[test]
    fn test_and_rd_rs1_rs2() -> Result<()> {
        let mut cpu = Cpu::new();
        let code = assemble(
            r#"
        addi x16, x0, 3
        addi x17, x0, 5
        and x18, x16, x17
        "#,
        )?;
        cpu.execute(&code)?;
        assert_eq!(cpu.reg[16], 3);
        assert_eq!(cpu.reg[17], 5);
        assert_eq!(cpu.reg[18], 1);
        Ok(())
    }

    #[test]
    fn test_lui_rd_imm() -> Result<()> {
        let mut cpu = Cpu::new();
        let code = assemble(
            r#"
        lui x16, 2
        "#,
        )?;
        cpu.execute(&code)?;
        assert_eq!(cpu.reg[16], 8192);
        Ok(())
    }

    #[test]
    fn test_beq_rs1_rs2_imm() -> Result<()> {
        let mut cpu = Cpu::new();
        let code = assemble(
            r#"
        addi x16, x0, 3
        addi x17, x0, 3
        beq x16, x17, 8
        addi x18, x0, 0    # This instruction will not be executed
        "#,
        )?;
        cpu.execute(&code)?;
        assert_eq!(cpu.reg[16], 3);
        assert_eq!(cpu.reg[17], 3);
        // Since the jump is taken, x18 should not be modified
        assert_eq!(cpu.reg[18], 0);
        Ok(())
    }

    #[test]
    fn test_bne_rs1_rs2_imm() -> Result<()> {
        let mut cpu = Cpu::new();
        let code = assemble(
            r#"
        addi x16, x0, 3
        addi x17, x0, 5
        bne x16, x17, 8
        addi x18, x0, 0    # This instruction will not be executed
        "#,
        )?;
        cpu.execute(&code)?;
        assert_eq!(cpu.reg[16], 3);
        assert_eq!(cpu.reg[17], 5);
        assert_eq!(cpu.reg[18], 0);
        Ok(())
    }

    #[test]
    fn test_jalr_rd_imm() -> Result<()> {
        let mut cpu = Cpu::new();
        cpu.reg[1] = 100; // Set x1 as the return address
        let code = assemble(
            r#"
        addi x16, x0, 3
        jalr x17, 4(x1)
        "#,
        )?;
        cpu.execute(&code)?;
        assert_eq!(cpu.reg[16], 3);
        assert_eq!(cpu.pc, 104); // Jump to x1 + 4
        assert_eq!(cpu.reg[17], cpu.pc + 4); // x17 saves the return address
        Ok(())
    }

    #[test]
    fn test_jal_rd_imm() -> Result<()> {
        let mut cpu = Cpu::new();
        cpu.pc = 100;
        let code = assemble(
            r#"
        jal x18, 8
        "#,
        )?;
        cpu.execute(&code)?;
        assert_eq!(cpu.reg[18], 104); // x18 saves the return address
        assert_eq!(cpu.pc, 108); // Jump to pc + 8
        Ok(())
    }
}
