.section .data
    .align 16
        inputs:
            .float 1.0, 2.0, 3.0, 4.0, 5.0, 6.0
        outputs:
            .float 0.5, 1.0, 1.5, 2.0, 2.5, 3.0
        size:
            .quad 6
        prediction:
            .float 17.0

.section .text
    fmt:
        .string "%f\n"

.global main
.extern printf

main:
    push %rbp
    mov %rsp, %rbp

    call train

    movss %xmm1, %xmm2
    movss %xmm0, %xmm1
    movss prediction(%rip), %xmm3

    call predict

    # Convert to double for printf
    cvtss2sd %xmm0, %xmm0

    lea fmt(%rip), %rdi
    mov $1, %eax
    call printf

    pop %rbp
    xor %eax, %eax
    ret


/*
* Performs linear regression on a dataset of n points.
* returns:
*        xmm0: slope (m)
*        xmm1: intercept (b)
*/
train:
    xorps %xmm11, %xmm11            # Clear denominator
    xorps %xmm12, %xmm12            # Clear x_sum
    xorps %xmm13, %xmm13            # Clear y_sum
    xorps %xmm14, %xmm14            # Clear xy_sum
    xorps %xmm15, %xmm15            # Clear x2_sum

    xorps %xmm0, %xmm0              # Clear slope (m)
    xorps %xmm1, %xmm1              # Clear intercept (b)

    xorps %xmm7, %xmm7              # Clear temp
    xorps %xmm8, %xmm8              # Clear temp2
    xor %rsi, %rsi                  # Clear loop counter (i)


    mov size(%rip), %rdi            # Load data size into rdi
  loop:
    movss inputs(,%rsi,4), %xmm7    # temp = inputs[i]
    addss %xmm7, %xmm12             # x_sum += inputs[i]

    movss outputs(,%rsi,4), %xmm7   # temp = outputs[i]
    addss %xmm7, %xmm13             # y_sum += outputs[i]

    movss inputs(,%rsi,4), %xmm7    # temp = inputs[i]
    movss outputs(,%rsi,4), %xmm8   # temp2 = outputs[i]
    mulss %xmm8, %xmm7              # temp = inputs[i] * outputs[i]
    movss %xmm7, %xmm14             # xy_sum += temp
    
    movss inputs(,%rsi,4), %xmm7    # temp = inputs[i]
    mulss %xmm7, %xmm7              # temp = inputs[i]^2
    movss %xmm7, %xmm15             # x2_sum += temp

    inc %rsi                        # i++
    cmp %rsi, %rdi                  # Compare i with size
    jl loop                         # Loop if i < size
  # End loop

    cvtsi2ss %rdi, %xmm11           # denominator = size
    mulss %xmm15, %xmm11            # denominator *= size

    movss %xmm12, %xmm7             # temp = x_sum
    mulss %xmm7, %xmm7              # temp = x_sum^2
    
    subss %xmm7, %xmm11             # denominator -= temp


    # Calculate slope
    cvtsi2ss %rdi, %xmm0           # slope = size
    mulss %xmm14, %xmm0            # slope *= xy_sum

    movss %xmm12, %xmm7            # temp = x_sum
    mulss %xmm13, %xmm7            # temp *= Y_sum

    subss %xmm7, %xmm0             # slope -= temp
    divss %xmm11, %xmm0            # slope /= denominator


    # Calculate Intercept
    movss %xmm13, %xmm1            # intercept = y_sum

    movss %xmm12, %xmm7            # temp = x_sum
    mulss %xmm0, %xmm7             # temp *= slope

    subss %xmm7, %xmm1             # intercept -= temp
    cvtsi2ss %rdi, %xmm7           # temp = size
    divss %xmm7, %xmm1             # intercept /= size

    ret
    

/*
* Uses a slope, intercept, and an x to predict a y value.
* xmm1: slope (m)
* xmm2: intercept (b)
* xmm3: x value
* returns:
*        xmm0: predicted y value
*/
predict:
    movss %xmm1, %xmm0             # predicted_y = slope
    mulss %xmm3, %xmm0             # predicted_y *= x
    addss %xmm2, %xmm0             # predicted_y += intercept

    ret
