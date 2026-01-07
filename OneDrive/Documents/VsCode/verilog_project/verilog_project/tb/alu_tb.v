`timescale 1ns/1ps
`default_nettype none

module alu #(
    parameter integer WIDTH = 32
)(
    input  wire [WIDTH-1:0] a,
    input  wire [WIDTH-1:0] b,
    input  wire [3:0]       op,
    output reg  [WIDTH-1:0] out
);

    // 1 = subtract, 0 = normal add (only used if you add an ADD op later)
    wire sub = (op == 4'b0001);

    // Two's complement trick: A - B = A + (~B + 1)
    wire [WIDTH-1:0] b_mod = b ^ {WIDTH{sub}};
    wire [WIDTH-1:0] sum   = a + b_mod + sub;

    always @(*) begin
        case (op)
            4'b0000: out = a & b;          // AND
            4'b0001: out = sum;            // SUB
            4'b0010: out = a | b;          // OR
            4'b0011: out = a ^ b;          // XOR
            4'b0100: out = ~a;             // NOT A
            4'b0101: out = a << 1;         // SHIFT LEFT
            4'b0110: out = a >> 1;         // SHIFT RIGHT
            default: out = {WIDTH{1'b0}};  // DEFAULT ZERO
        endcase
    end

endmodule

`default_nettype wire

