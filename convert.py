import struct


class Convert:
    def __init__(self, n_bits):
        self.n_bits = n_bits
        self.L = n_bits * 8  # Total number of bits

        # Determine struct format based on bit size
        if self.L == 16:
            self.format_char = 'e'  # Float16
        elif self.L == 32:
            self.format_char = 'f'  # Float32
        else:
            raise ValueError("Unsupported bit size. Use 16-bit or 32-bit floats.")

    def floatToBits(self, f):
        """Convert float to bit representation as an integer."""
        s = struct.pack(f'>{self.format_char}', f)
        return int.from_bytes(s, byteorder='big')

    def bitsToFloat(self, b):
        """Convert integer bit representation back to float."""
        s = b.to_bytes(self.n_bits, byteorder='big')
        return struct.unpack(f'>{self.format_char}', s)[0]

    def get_bits(self, x):
        """Convert float to binary string representation."""
        x = self.floatToBits(x)
        return format(x, f'0{self.L}b')  # Convert to binary with leading zeros

    def get_float(self, bits):
        """Convert binary string back to float."""
        assert len(bits) == self.L, "Invalid bit string length"
        x = int(bits, 2)  # Convert binary string to integer
        return self.bitsToFloat(x)
