use crate::cpu::Cpu;
use super::{format::{UFormat, IFormat},InsImpl};
use std::{sync::LazyLock, collections::BTreeMap};

pub static OPCODE2INS_IMPL: LazyLock<BTreeMap<u8, InsImpl>> = LazyLock::new(|| {
    let mut map = BTreeMap::new();
    register_instructions(&mut map);
    map
});

pub fn lui(instruction: u32, cpu: &mut Cpu) {
    let f = UFormat(instruction);
    let imm = f.imm();
    cpu.reg[f.rd() as usize] = sext(imm << 12, 31);
}

pub fn auipc(instruction: u32, cpu: &mut Cpu) {
    let f = UFormat(instruction);
    let imm = f.imm();
    cpu.reg[f.rd() as usize] = cpu.pc + sext(imm << 12, 31);
}

pub fn addi(instruction: u32, cpu: &mut Cpu) {
    let f = IFormat(instruction);
    let rd = f.rd() as usize;
    let rs1 = f.rs1() as usize;
    let imm = sext(f.imm(), 11);
    cpu.reg[rd] = cpu.reg[rs1].wrapping_add(imm);
}

pub fn l(instruction: u32, cpu: &mut Cpu) {
    let f = IFormat(instruction);
    match f.funct3() {
        0b000 => load_mem::<1>(f, cpu), // lb
        0b001 => load_mem::<2>(f, cpu), // lh
        0b010 => load_mem::<4>(f, cpu), // lw
        _ => panic!("valid load instrunction func4 0b{:b}", f.funct3()),
    }
    
}

fn load_mem<const BYTES: usize>(instruction: IFormat, cpu: &mut Cpu) {
    let f = instruction;
    let offset = sext(f.imm(), 11);
    let rs1 = f.rs1() as usize;
    let rd = f.rd() as usize;
    let addr = cpu.reg[rs1] + offset;
    let mut buf = [0; std::mem::size_of::<u32>()];
    cpu.bus.read(addr, &mut buf[0..BYTES]);
    let value = u32::from_le_bytes(buf);
    let msb = (8 << BYTES) - 1;
    cpu.reg[rd] = sext(value, msb);
}

fn register_instructions(map: &mut BTreeMap<u8, InsImpl>) {
    map.insert(0b01101, lui);
    map.insert(0b00101, auipc);
    map.insert(0b00100, addi);
    map.insert(0b00000, l);
}

fn sext(imm: u32, msb: usize) -> u64 {
    let m = 1u64.wrapping_shl(msb as u32);
    let imm = imm as u64;
    if imm & m > 0 {
        (!(m-1 | m)) | imm
    } else {
        imm
    }
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
            color_eyre::eyre::bail!("fail to call riscv64-unknown-linux-gnu-as");
        }
        Ok(())
    }

    fn assemble(code: &str) -> Result<Vec<u32>> {
        let mut source = tempfile::Builder::new().suffix(".s").tempfile()?;
        let mut obj_path = source.path().to_owned();
        obj_path.set_extension("o");
        source.write_all(code.as_bytes())?;
        source.write_all(b"\n")?;
        call_external_command(Command::new("/opt/riscv/bin/riscv64-unknown-linux-gnu-as").arg(source.path()).arg("-o").arg(&obj_path))?;
        let obj_bin = std::fs::read(&obj_path)?;
        let obj_file = object::read::File::parse(&*obj_bin)?;
        let text_sec = obj_file.section_by_name(".text").wrap_err("no .text section found")?;
        let data = text_sec.data()?;
        let ins_buf: Result<Vec<_>> = data.chunks_exact(4).map(|b| Ok(u32::from_le_bytes(b.try_into()?))).collect();
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

        Ok(())
    }

    #[test]
    fn test_l() -> Result<()> {
        tracing_subscriber::fmt::init();
        let mut cpu = Cpu::new();
        cpu.bus.write(4, &[0x12, 0x34, 0x56, 0x78]);
        let code = assemble(r#"
            addi x16, x0, 5
            addi x17, x0, 3
            lb x18, 4(x0)
            lh x19, 4(x0)
            lw x20, 4(x0)
        "#)?;
        cpu.execute(&code)?;
        assert_eq!(cpu.reg[16], 5);
        assert_eq!(cpu.reg[17], 3);
        assert_eq!(cpu.reg[18], 0x12);
        assert_eq!(cpu.reg[19], 0x3412);
        assert_eq!(cpu.reg[20], 0x78563412);
        Ok(())
    }
}