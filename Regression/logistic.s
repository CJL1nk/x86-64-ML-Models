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

        input_height:
            .float 72.0
        input_weight:
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


    xorps %xmm0, %xmm0              # Clear w1
    xorps %xmm1, %xmm1              # Clear w2
    movss e(%rip), %xmm1            # Load 2.71 into xmm1

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
 
    addss %xmm2, %xmm0               # Compute 1 + exp(-z)
    divss %xmm0, %xmm2               # Compute 1 / (1 + exp(-z))
    movss %xmm2, %xmm0               # Move result to xmm0 for return

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

    # Model parameters
    xorps %xmm0, %xmm0              # Clear w1
    xorps %xmm1, %xmm1              # Clear w2
    xorps %xmm2, %xmm2              # Clear b


    xor %rsi, %rsi                  # Clear loop counter (i)
    mov epochs(%rip), %rdi          # Load epochs into rdi
  loop1:
      
      xorps %xmm13, %xmm13          # Clear dw1
      xorps %xmm14, %xmm14          # Clear dw2
      xorps %xmm15, %xmm15          # Clear db

      push %rsi                     # Save loop counter for inner loop
      push %rdi                     # Save epochs for inner loop


      mov size(%rip), %rdi          # Load data size into rdi
      xor %rsi, %rsi                # Clear loop counter for inner loop
    loop2:
        xorps %xmm12, %xmm12        # Clear z               (will also hold p AND error in future)
        xorps %xmm9, %xmm9          # Clear temp1
        xorps %xmm10, %xmm10        # Clear temp2

        movss heights(,%rsi,4), %xmm9  # Load height[i]
        movss weights(,%rsi,4), %xmm10 # Load weight[i]

        mulss %xmm0, %xmm9          # w1 *= height[i]
        mulss %xmm1, %xmm10         # w2 *= weight[i]

        addss %xmm9, %xmm12         # z += w1 * height[i]
        addss %xmm10, %xmm12        # z += w2 * weight[i]
        addss %xmm2, %xmm12         # z += b

        subq $16, %rsp              # Push xmm0 (w1) before sigmoid 
        movups %xmm0, (%rsp)
        subq $16, %rsp              # Push xmm1 (w2) before sigmoid
        movups %xmm1, (%rsp)
        movss %xmm12, %xmm1         # Move z into xmm1 for sigmoid

        call sigmoid                # p = sigmoid(z)        (p to be held in xmm12)
        movss %xmm0 , %xmm12

        movups (%rsp), %xmm1        # Restore w2
        addq $16, %rsp
        movups (%rsp), %xmm0        # Restore w1
        addq $16, %rsp
      
        subss outputs(,%rsi,4), %xmm12 # error = p - y[i]   (error to be held in xmm12)

        movss %xmm12, %xmm9         # temp1 = error
        mulss heights(,%rsi,4), %xmm9 # temp1 *= height[i]
        addss %xmm9, %xmm13         # dw1 += temp1

        movss %xmm12, %xmm10        # temp2 = error
        mulss weights(,%rsi,4), %xmm10 # temp2 *= weight[i]
        addss %xmm10, %xmm14        # dw2 += temp2

        addss %xmm12, %xmm15        # db += error

        inc %rsi                    # i++
        cmp %rsi, %rdi              # Compare i with size
        jl loop2                    # Loop if i < size
    # End loop2
    
      pop %rdi                       # Restore epochs
      pop %rsi                       # Restore loop counter

      divss size(%rip), %xmm13       # dw1 /= size
      divss size(%rip), %xmm14       # dw2 /= size
      divss size(%rip), %xmm15       # db /= size

      movss learning_rate(%rip), %xmm9 # temp1 = learning_rate
      mulss %xmm13, %xmm9            # temp1 *= dw1
      subss %xmm9, %xmm0             # w1 -= temp1

      movss learning_rate(%rip), %xmm9 # temp1 = learning_rate
      mulss %xmm14, %xmm9            # temp1 *= dw2
      subss %xmm9, %xmm1             # w2 -= temp1

      movss learning_rate(%rip), %xmm9 # temp1 = learning_rate
      mulss %xmm15, %xmm9            # temp1 *= db
      subss %xmm9, %xmm2             # b -= temp1

      inc %rsi                       # epoch++
      cmp %rsi, %rdi                 # Compare epoch with epochs
      jl loop1                       # Loop if epoch < epochs
  # End loop1

    ret


/**
* Predicts probabilities for a 2D logistic regression model
* inputs:
*        xmm1: w1
*        xmm2: w2
*        xmm3: b
*        xmm4: input_height
*        xmm5: input_weight
* returns:
*        xmm0: predicted probability
*/

predict_prob_2d:

    


predict_class_2d:


