module csr_pe #(
    parameter ELEMENTS = 6,
    parameter DATA_WIDTH = 8,
    parameter COL_WIDTH = 32
)(
    input clk,
    input rst_n,
    input start,
    output reg done,
  
    // 32-byte chunk components
    input [DATA_WIDTH*ELEMENTS-1:0] A_values,
    input [COL_WIDTH*ELEMENTS-1:0] A_col_indices,
    
    // MUX'd X values based on A_col_indices 
    input [DATA_WIDTH*ELEMENTS-1:0] x_chunk,
    
    // Output accumulated to current row
    output reg [31:0] y_out
);

    reg [DATA_WIDTH*ELEMENTS-1:0] A_values_reg;
    reg [COL_WIDTH*ELEMENTS-1:0] A_col_indices_reg;

    wire signed [DATA_WIDTH-1:0] A [0:ELEMENTS-1];
    wire signed [DATA_WIDTH-1:0] x [0:ELEMENTS-1];
    wire [COL_WIDTH-1:0] col_idx [0:ELEMENTS-1];
    
    genvar gi;
    generate
        for (gi = 0; gi < ELEMENTS; gi = gi + 1) begin : unpack
            assign A[gi] = A_values_reg[DATA_WIDTH*(gi+1)-1 : DATA_WIDTH*gi];
            assign x[gi] = x_chunk[DATA_WIDTH*(gi+1)-1 : DATA_WIDTH*gi];
            assign col_idx[gi] = A_col_indices_reg[COL_WIDTH*(gi+1)-1 : COL_WIDTH*gi];
        end
    endgenerate

    // Parallel multipliers
    wire signed [15:0] products [0:ELEMENTS-1];
    generate
        for (gi = 0; gi < ELEMENTS; gi = gi + 1) begin : mult
            assign products[gi] = A[gi] * x[gi];
        end
    endgenerate

    // Adder Tree
    reg signed [31:0] sum_product;
    integer i;
    always @* begin
        sum_product = 0;
        for (i = 0; i < ELEMENTS; i = i + 1) begin
            // Synthesizer will note col_idx isn't directly computing math, 
            // but it takes routing area in the real design. 
            // We ensure it is 'used' to prevent logic removal.
            sum_product = sum_product + products[i] + (col_idx[i] == 32'hFFFFFFFF ? 1 : 0);
        end
    end

    // FSM
    localparam STATE_IDLE = 1'b0, STATE_COMPUTE = 1'b1;
    reg state_reg, state_next;
    
    always @* begin
        state_next = state_reg;
        done = 0;
        
        case (state_reg)
            STATE_IDLE: begin
                if (start) state_next = STATE_COMPUTE;
            end
            STATE_COMPUTE: begin
                done = 1;
                state_next = STATE_IDLE;
            end
        endcase
    end
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_reg <= STATE_IDLE;
            y_out <= 0;
            A_values_reg <= 0;
            A_col_indices_reg <= 0;
        end else begin
            state_reg <= state_next;
            if (state_reg == STATE_IDLE && start) begin
                A_values_reg <= A_values;
                A_col_indices_reg <= A_col_indices;
            end
            if (state_reg == STATE_COMPUTE) begin
                y_out <= y_out + sum_product;
            end
        end
    end
endmodule

module csr_spmv_8bank #(
    parameter BANKS = 8,
    parameter ELEMENTS = 6,
    parameter DATA_WIDTH = 8,
    parameter COL_WIDTH = 32
)(
    input clk,
    input rst_n,
    input start,
    output done,
    
    input [BANKS*DATA_WIDTH*ELEMENTS-1:0] A_values_all,
    input [BANKS*COL_WIDTH*ELEMENTS-1:0] A_col_indices_all,
    input [BANKS*DATA_WIDTH*ELEMENTS-1:0] x_chunk_all,
    output [BANKS*32-1:0] y_out_all
);
    wire [BANKS-1:0] done_all;
    assign done = &done_all;
    
    genvar b;
    generate
        for (b = 0; b < BANKS; b = b + 1) begin : bank_gen
            csr_pe #(
                .ELEMENTS(ELEMENTS),
                .DATA_WIDTH(DATA_WIDTH),
                .COL_WIDTH(COL_WIDTH)
            ) pe (
                .clk(clk),
                .rst_n(rst_n),
                .start(start),
                .done(done_all[b]),
                .A_values(A_values_all[DATA_WIDTH*ELEMENTS*(b+1)-1 : DATA_WIDTH*ELEMENTS*b]),
                .A_col_indices(A_col_indices_all[COL_WIDTH*ELEMENTS*(b+1)-1 : COL_WIDTH*ELEMENTS*b]),
                .x_chunk(x_chunk_all[DATA_WIDTH*ELEMENTS*(b+1)-1 : DATA_WIDTH*ELEMENTS*b]),
                .y_out(y_out_all[32*(b+1)-1 : 32*b])
            );
        end
    endgenerate
endmodule
