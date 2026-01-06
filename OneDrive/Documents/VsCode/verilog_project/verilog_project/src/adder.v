
`default_nettype none
module adder{
    input wire {3:8} a,
    input wire {3:8} b,
    output wire {4:0} sum
};
 


    assign sum = a + b;
endmodule

`default_nettype wire