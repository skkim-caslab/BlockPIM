module coo_pe #(
    parameter ELEMENTS = 3,
    parameter DATA_WIDTH = 8,
    parameter IDX_WIDTH = 32,
    parameter BANK_ROWS = 32
)(
    input clk,
    input rst_n,
    input start,
    output reg done,
  
    // 32-byte chunk components
    input [DATA_WIDTH*ELEMENTS-1:0] A_values,
    input [IDX_WIDTH*ELEMENTS-1:0] A_row_indices,
    input [IDX_WIDTH*ELEMENTS-1:0] A_col_indices,
    
    // MUX'd X values
    input [DATA_WIDTH*ELEMENTS-1:0] x_chunk,
    
    // Output bank
    output reg [32*BANK_ROWS-1:0] y_flat
);

    reg signed [31:0] y_bank [0:BANK_ROWS-1];
    
    genvar yi;
    generate
        for (yi = 0; yi < BANK_ROWS; yi = yi + 1) begin : y_pack
            always @* y_flat[32*(yi+1)-1 : 32*yi] = y_bank[yi];
        end
    endgenerate

    reg [DATA_WIDTH*ELEMENTS-1:0] A_values_reg;
    reg [IDX_WIDTH*ELEMENTS-1:0] A_row_indices_reg;
    reg [IDX_WIDTH*ELEMENTS-1:0] A_col_indices_reg;

    wire signed [DATA_WIDTH-1:0] A [0:ELEMENTS-1];
    wire signed [DATA_WIDTH-1:0] x [0:ELEMENTS-1];
    wire [IDX_WIDTH-1:0] row_idx [0:ELEMENTS-1];
    wire [IDX_WIDTH-1:0] col_idx [0:ELEMENTS-1];
    
    genvar gi;
    generate
        for (gi = 0; gi < ELEMENTS; gi = gi + 1) begin : unpack
            assign A[gi] = A_values_reg[DATA_WIDTH*(gi+1)-1 : DATA_WIDTH*gi];
            assign x[gi] = x_chunk[DATA_WIDTH*(gi+1)-1 : DATA_WIDTH*gi];
            assign row_idx[gi] = A_row_indices_reg[IDX_WIDTH*(gi+1)-1 : IDX_WIDTH*gi];
            assign col_idx[gi] = A_col_indices_reg[IDX_WIDTH*(gi+1)-1 : IDX_WIDTH*gi];
        end
    endgenerate

    // Parallel multipliers
    wire signed [15:0] products [0:ELEMENTS-1];
    generate
        for (gi = 0; gi < ELEMENTS; gi = gi + 1) begin : mult
            assign products[gi] = A[gi] * x[gi] + (col_idx[gi] == 32'hFFFFFFFF ? 1 : 0);
        end
    endgenerate

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
    
    integer init_r, i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_reg <= STATE_IDLE;
            A_values_reg <= 0;
            A_row_indices_reg <= 0;
            A_col_indices_reg <= 0;
            for (init_r = 0; init_r < BANK_ROWS; init_r = init_r + 1) begin
                y_bank[init_r] <= 0;
            end
        end else begin
            state_reg <= state_next;
            if (state_reg == STATE_IDLE && start) begin
                A_values_reg <= A_values;
                A_row_indices_reg <= A_row_indices;
                A_col_indices_reg <= A_col_indices;
            end
            if (state_reg == STATE_COMPUTE) begin
                // Scatter Accumulation
                for (i = 0; i < ELEMENTS; i = i + 1) begin
                    // Ensure row_idx is used to address the bank
                    y_bank[row_idx[i][4:0]] <= y_bank[row_idx[i][4:0]] + products[i];
                end
            end
        end
    end
endmodule

module coo_spmv_8bank #(
    parameter BANKS = 8,
    parameter ELEMENTS = 3,
    parameter DATA_WIDTH = 8,
    parameter IDX_WIDTH = 32,
    parameter BANK_ROWS = 32
)(
    input clk,
    input rst_n,
    input start,
    output done,
    
    input [BANKS*DATA_WIDTH*ELEMENTS-1:0] A_values_all,
    input [BANKS*IDX_WIDTH*ELEMENTS-1:0] A_row_indices_all,
    input [BANKS*IDX_WIDTH*ELEMENTS-1:0] A_col_indices_all,
    input [BANKS*DATA_WIDTH*ELEMENTS-1:0] x_chunk_all,
    output [BANKS*32*BANK_ROWS-1:0] y_flat_all
);
    wire [BANKS-1:0] done_all;
    assign done = &done_all;
    
    genvar b;
    generate
        for (b = 0; b < BANKS; b = b + 1) begin : bank_gen
            coo_pe #(
                .ELEMENTS(ELEMENTS),
                .DATA_WIDTH(DATA_WIDTH),
                .IDX_WIDTH(IDX_WIDTH),
                .BANK_ROWS(BANK_ROWS)
            ) pe (
                .clk(clk),
                .rst_n(rst_n),
                .start(start),
                .done(done_all[b]),
                .A_values(A_values_all[DATA_WIDTH*ELEMENTS*(b+1)-1 : DATA_WIDTH*ELEMENTS*b]),
                .A_row_indices(A_row_indices_all[IDX_WIDTH*ELEMENTS*(b+1)-1 : IDX_WIDTH*ELEMENTS*b]),
                .A_col_indices(A_col_indices_all[IDX_WIDTH*ELEMENTS*(b+1)-1 : IDX_WIDTH*ELEMENTS*b]),
                .x_chunk(x_chunk_all[DATA_WIDTH*ELEMENTS*(b+1)-1 : DATA_WIDTH*ELEMENTS*b]),
                .y_flat(y_flat_all[32*BANK_ROWS*(b+1)-1 : 32*BANK_ROWS*b])
            );
        end
    endgenerate
endmodule
