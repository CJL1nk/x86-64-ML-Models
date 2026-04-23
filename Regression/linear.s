.section .data
    .align 16
        inputs:
            .float 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0

        outputs:
            .float 0.6, 0.9, 1.6, 2.1, 2.4, 3.2, 3.6, 4.1, 4.6, 5.1, 5.7, 6.2, 6.4, 7.3, 7.6, 8.0, 8.6, 9.1, 9.4, 10.2, 10.6, 11.1, 11.7, 12.0, 12.6, 13.2, 13.4, 14.1, 14.6, 15.2,  16.1, 16.4, 17.2, 17.6, 18.0, 18.7, 19.1, 19.6, 20.2

        size:
            .quad 40
        prediction:
            .float 6.0

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
    cvtsi2ss %rdi, %xmm0            # slope = size
    mulss %xmm14, %xmm0             # slope *= xy_sum

    movss %xmm12, %xmm7             # temp = x_sum
    mulss %xmm13, %xmm7             # temp *= Y_sum

    subss %xmm7, %xmm0              # slope -= temp
    divss %xmm11, %xmm0             # slope /= denominator


    # Calculate Intercept
    movss %xmm13, %xmm1             # intercept = y_sum

    movss %xmm12, %xmm7             # temp = x_sum
    mulss %xmm0, %xmm7              # temp *= slope

    subss %xmm7, %xmm1              # intercept -= temp
    cvtsi2ss %rdi, %xmm7            # temp = size
    divss %xmm7, %xmm1              # intercept /= size

    ret
    

/*
* Uses a slope, intercept, and an x to predict a y value.
*
* inputs:
*        xmm1: slope (m)
*        xmm2: intercept (b)
*        xmm3: x value
* returns:
*        xmm0: predicted y value
*/
predict:
    movss %xmm1, %xmm0              # predicted_y = slope
    mulss %xmm3, %xmm0              # predicted_y *= x
    addss %xmm2, %xmm0              # predicted_y += intercept

    ret
