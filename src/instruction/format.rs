use super::{low_bitmask_u32, low_bitmask_u8};

// | 31 - 25 | 24 - 20 | 19 - 15 |  14 - 12 | 11 - 7 | 6 - 0  |
// | funct7  |   rs2   |   rs1   |  funct3  |   rd   | opcode |
#[derive(Clone, Copy)]
pub struct RFormat(pub u32);
impl RFormat {
    pub fn funct7(self) -> u32 {
        self.0 >> 25
    }

    pub fn rs2(self) -> u8 {
        ((self.0 >> 20) & low_bitmask_u32(5)) as u8
    }

    pub fn rs1(self) -> u8 {
        ((self.0 >> 15) & low_bitmask_u32(5)) as u8
    }

    pub fn funct3(self) -> u8 {
        ((self.0 >> 12) & low_bitmask_u32(3)) as u8
    }

    pub fn rd(self) -> u8 {
        ((self.0 >> 7) & low_bitmask_u32(5)) as u8
    }

    pub fn opcode(self) -> u8 {
        (self.0 & low_bitmask_u32(7)) as u8
    }
}

// |  31 - 20  | 19 - 15 |  14 - 12 | 11 - 7 | 6 - 0  |
// | imm[11:0] |   rs1   |  funct3  |   rd   | opcode |
#[derive(Clone, Copy)]
pub struct IFormat(pub u32);
impl IFormat {
    pub fn imm(self) -> u32 {
        self.0 >> 20
    }

    pub fn rs1(self) -> u8 {
        ((self.0 >> 15) & low_bitmask_u32(5)) as u8
    }

    pub fn funct3(self) -> u8 {
        ((self.0 >> 12) & low_bitmask_u32(3)) as u8
    }

    pub fn rd(self) -> u8 {
        ((self.0 >> 7) & low_bitmask_u32(5)) as u8
    }

    pub fn opcode(self) -> u8 {
        (self.0 & low_bitmask_u32(7)) as u8
    }
}

// |  31 - 25  | 24 - 20 | 19 - 15 |  14 - 12 |  11 - 7  | 6 - 0  |
// | imm[11:5] |   rs2   |   rs1   |  funct3  | imm[4:0] | opcode |
#[derive(Clone, Copy)]
pub struct SFormat(pub u32);
impl SFormat {
    pub fn imm(self) -> u32 {
        ((self.0 >> 25) << 5) | ((self.0 >> 7) & low_bitmask_u32(5))
    }

    pub fn rs2(self) -> u8 {
        ((self.0 >> 20) & low_bitmask_u32(5)) as u8
    }

    pub fn rs1(self) -> u8 {
        ((self.0 >> 15) & low_bitmask_u32(5)) as u8
    }

    pub fn funct3(self) -> u8 {
        ((self.0 >> 12) & low_bitmask_u32(3)) as u8
    }

    pub fn opcode(self) -> u8 {
        (self.0 & low_bitmask_u32(7)) as u8
    }
}

// |   31 - 25    | 24 - 20 | 19 - 15 |  14 - 12 |    11 - 7   | 6 - 0  |
// | imm[12|10:5] |   rs2   |   rs1   |  funct3  | imm[4:1|11] | opcode |
#[derive(Clone, Copy)]
pub struct SBFormat(pub u32);
impl SBFormat {
    pub fn imm(self) -> u32 {
        let bit12 = self.0 >> 31;
        let bit10_5 = (self.0 >> 25) & low_bitmask_u32(6);
        let bit4_1 = (self.0 >> 8) & low_bitmask_u32(4);
        let bit11 = (self.0 >> 7) & 0b1;
        (bit12 << 12) | (bit11 << 11) | (bit10_5 << 5) | (bit4_1 << 1)
    }

    pub fn rs2(self) -> u8 {
        ((self.0 >> 20) & low_bitmask_u32(5)) as u8
    }

    pub fn rs1(self) -> u8 {
        ((self.0 >> 15) & low_bitmask_u32(5)) as u8
    }

    pub fn funct3(self) -> u8 {
        ((self.0 >> 12) & low_bitmask_u32(3)) as u8
    }

    pub fn opcode(self) -> u8 {
        (self.0 & low_bitmask_u32(7)) as u8
    }
}

// |   31 - 12   | 11 - 7 | 6 - 0  |
// |  imm[31:12] |   rd   | opcode |
#[derive(Clone, Copy)]
pub struct UFormat(pub u32);
impl UFormat {
    pub fn imm(self) -> u32 {
        self.0 >> 12
    }

    pub fn rd(self) -> u8 {
        ((self.0 >> 7) & low_bitmask_u32(5)) as u8
    }

    pub fn opcode(self) -> u8 {
        (self.0 & low_bitmask_u32(7)) as u8
    }
}

// |         31 - 12        | 11 - 7 | 6 - 0  |
// |  imm[20|10:1|11|19:12] |   rd   | opcode |
#[derive(Clone, Copy)]
pub struct UJFormat(pub u32);
impl UJFormat {
    pub fn imm(self) -> u32 {
        let bit20 = self.0 >> 31;
        let bit10_1 = (self.0 >> 21) & low_bitmask_u32(10);
        let bit11 = (self.0 >> 19) & 0b1;
        let bit19_12 = (self.0 >> 12) & low_bitmask_u32(8);
        (bit20 << 20) | (bit19_12 << 12) | (bit11 << 11) | (bit10_1 << 1)
    }

    pub fn rd(self) -> u8 {
        ((self.0 >> 7) & low_bitmask_u32(5)) as u8
    }

    pub fn opcode(self) -> u8 {
        (self.0 & low_bitmask_u32(7)) as u8
    }
}
