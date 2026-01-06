`timescale 1ns/1ps
`default_nettype none

module counter_tb;

    reg clk;
    reg reset;
    wire [3:0] count;

    counter uut (
        .clk(clk),
        .reset(reset),
        .count(count)
    );

    // 100 MHz clock (10ns period)
    always #5 clk = ~clk;

initial begin
    // Tell simulator to write waveform file
    $dumpfile("counter.vcd");
    $dumpvars(0, counter_tb);

    clk = 0;
    reset = 1;

    #20 reset = 0;
    #100 $finish;
end


endmodule

`default_nettype wire
