.section .data
    .align 16
        e:
            .float 2.718281828459045
        one:
            .float 1.0
        negone:
            .float -1.0
        heights:
            .float 60.0, 62.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 61.0, 63.0, 67.0, 68.0, 72.0, 74.0, 76.0, 69.0, 71.0, 73.0, 75.0, 77.0, 79.0

        weights:
            .float 100.0, 110.0, 120.0, 125.0, 140.0, 150.0, 155.0, 160.0, 165.0, 175.0, 180.0, 190.0, 200.0, 210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 180.0, 200.0, 220.0, 240.0, 260.0, 280.0, 300.0, 210.0, 230.0, 250.0, 270.0, 290.0, 310.0

        outputs:
            .quad 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1

        size:
            .quad 32
        learning_rate:
            .float 0.01
        epochs:
            .quad 1000

        input1:
            .float 72.0
        input2:
            .float 250.0


.section .text
    fmt:
        .string "%f\n"

.global main
.extern printf
.extern expf


main:
    push %rbp
    mov %rsp, %rbp

    movss one(%rip), %xmm2          # Load 1.0 into xmm2

    xorps %xmm0, %xmm0              # Clear w1
    xorps %xmm1, %xmm1              # Clear w2
    addss %xmm2, %xmm1

    call sigmoid

    # Convert to double for printf
    cvtss2sd %xmm0, %xmm0
    lea fmt(%rip), %rdi
    mov $1, %eax
    call printf

    pop %rbp
    xor %eax, %eax
    ret


/*
* Sigmoid function for logistic regression
* inputs:
*        xmm1: z (the linear combination of inputs and weights)
* returns:
*        xmm0: sigmoid(z)
*/
sigmoid:

    subq $16, %rsp
    movups %xmm2, (%rsp)

    movss negone(%rip), %xmm2        # Load -1.0 into xmm2

    movss %xmm1, %xmm0               # Move z into xmm0 for computation
    mulss %xmm2, %xmm0               # Negate z for exp(-z)
    call expf                        # Compute exp(-z)

    movss one(%rip), %xmm2           # Load 1.0 into xmm2
 
    addss %xmm2, %xmm0               # Compute 1 + exp
    divss %xmm2, %xmm0               # Compute 1 / (1 + exp(-z))

    movups (%rsp), %xmm2
    addq $16, %rsp
    ret


/*
* Trains a logistic regression model based on a dataset defined in the data segment
* GRADIENT DESCENT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
* returns:
*        xmm0: w1
*        xmm1: w2
*        xmm2: b
*/
train_logistic_2d:

    xorps %xmm13, %xmm13            # Clear dw1
    xorps %xmm14, %xmm14            # Clear dw2
    xorps %xmm15, %xmm15            # Clear db

    xorps %xmm0, %xmm0              # Clear w1
    xorps %xmm1, %xmm1              # Clear w2
    xorps %xmm2, %xmm2              # Clear b

    xor %rsi, %rsi                  # Clear loop counter (i)


    mov size(%rip), %rdi            # Load data size into rdi
  loop:
    


predict_prob_2d:


predict_class_2d:


