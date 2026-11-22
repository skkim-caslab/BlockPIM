// 1. dense_pe: processing element for 32 bytes
module dense_pe #(
    parameter PE_ELEMENTS = 32,
    parameter DATA_WIDTH = 8
)(
    input clk,
    input rst_n,
    input start,
    output reg done,
  
    input [DATA_WIDTH*PE_ELEMENTS-1:0] A_chunk,
    input [DATA_WIDTH*PE_ELEMENTS-1:0] x_chunk,
    output reg [31:0] grf_a,
    output reg [255:0] grf_b
);
    wire signed [DATA_WIDTH-1:0] A [0:PE_ELEMENTS-1];
    wire signed [DATA_WIDTH-1:0] x [0:PE_ELEMENTS-1];
    
    genvar gi;
    generate
        for (gi = 0; gi < PE_ELEMENTS; gi = gi + 1) begin : unpack
            assign A[gi] = A_chunk[DATA_WIDTH*(gi+1)-1 : DATA_WIDTH*gi];
            assign x[gi] = x_chunk[DATA_WIDTH*(gi+1)-1 : DATA_WIDTH*gi];
        end
    endgenerate

    wire signed [15:0] products [0:PE_ELEMENTS-1];
    generate
        for (gi = 0; gi < PE_ELEMENTS; gi = gi + 1) begin : mult
            assign products[gi] = A[gi] * x[gi];
        end
    endgenerate

    reg signed [31:0] sum_product;
    integer i;
    always @* begin
        sum_product = 0;
        for (i = 0; i < PE_ELEMENTS; i = i + 1) begin
            sum_product = sum_product + products[i];
        end
    end

    localparam STATE_IDLE = 1'b0, STATE_COMPUTE = 1'b1;
    reg state_reg, state_next;
    
    always @* begin
        state_next = state_reg;
        done = 0;
        if (state_reg == STATE_IDLE && start) state_next = STATE_COMPUTE;
        else if (state_reg == STATE_COMPUTE) begin
            done = 1;
            state_next = STATE_IDLE;
        end
    end
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_reg <= STATE_IDLE;
            grf_a <= 0;
            grf_b <= 0;
        end else begin
            state_reg <= state_next;
            if (state_reg == STATE_IDLE && start) begin
                grf_a <= A_chunk[31:0]; 
            end
            if (state_reg == STATE_COMPUTE) begin
                grf_b <= grf_b + {224'b0, sum_product};
            end
        end
    end
endmodule

// 2. bitmap_ffo: MUX and Find-First-One logic
module bitmap_ffo (
    input [255:0] full_bitmap,
    input [2:0] chunk_idx,
    output reg [4:0] dest_row
);
    wire [31:0] selected_bitmap;
    assign selected_bitmap = full_bitmap[32*chunk_idx +: 32];
    
    integer bi;
    always @* begin
        dest_row = 0;
        for (bi = 0; bi < 32; bi = bi + 1) begin
            if (selected_bitmap[bi] == 1'b1) begin
                dest_row = bi[4:0];
            end
        end
    end
endmodule

// 3. Top level: 8-Bank StackPIM architecture
module dense_stackpim_spmv_8bank #(
    parameter BANKS = 8,
    parameter ELEMENTS = 32, // Block size in bytes
    parameter DATA_WIDTH = 8
)(
    input clk,
    input rst_n,
    input start,
    output done,
    
    input [255:0] full_bitmap,
    input [BANKS*3-1:0] chunk_idx_all,
    input [BANKS*DATA_WIDTH*32-1:0] A_chunk_all,
    input [BANKS*DATA_WIDTH*32-1:0] x_chunk_all,
    
    output [BANKS*32-1:0] grf_a_all,
    output [BANKS*256-1:0] grf_b_all,
    output [BANKS*5-1:0] row_idx_out_all
);

    wire [BANKS-1:0] done_all;
    assign done = &done_all;
    
    localparam BANKS_PER_BLOCK = (ELEMENTS >= 32) ? (ELEMENTS / 32) : 1;
    localparam NUM_FFOS = (BANKS / BANKS_PER_BLOCK > 0) ? (BANKS / BANKS_PER_BLOCK) : 1;
    
    wire [4:0] shared_dest_row [0:NUM_FFOS-1];
    
    genvar f;
    generate
        for (f = 0; f < NUM_FFOS; f = f + 1) begin : ffo_gen
            bitmap_ffo ffo_inst (
                .full_bitmap(full_bitmap),
                .chunk_idx(chunk_idx_all[3*(f * BANKS_PER_BLOCK) +: 3]),
                .dest_row(shared_dest_row[f])
            );
        end
    endgenerate
    
    genvar b;
    generate
        for (b = 0; b < BANKS; b = b + 1) begin : bank_gen
            localparam FFO_IDX = (b / BANKS_PER_BLOCK < NUM_FFOS) ? (b / BANKS_PER_BLOCK) : (NUM_FFOS - 1);
            
            assign row_idx_out_all[5*b +: 5] = shared_dest_row[FFO_IDX];
            
            dense_pe #(
                .PE_ELEMENTS(32),
                .DATA_WIDTH(DATA_WIDTH)
            ) pe (
                .clk(clk),
                .rst_n(rst_n),
                .start(start),
                .done(done_all[b]),
                .A_chunk(A_chunk_all[DATA_WIDTH*32*b +: DATA_WIDTH*32]),
                .x_chunk(x_chunk_all[DATA_WIDTH*32*b +: DATA_WIDTH*32]),
                .grf_a(grf_a_all[32*b +: 32]),
                .grf_b(grf_b_all[256*b +: 256])
            );
        end
    endgenerate
endmodule
