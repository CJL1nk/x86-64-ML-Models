.section .data
    .align 16
        one:
            .double 1.0
        prob_threshold:
            .double 0.5
        negone:
            .double -1.0
        heights:
            .double 60.0, 62.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 61.0, 63.0, 67.0, 68.0, 72.0, 74.0, 76.0, 69.0, 71.0, 73.0, 75.0, 77.0, 79.0

        weights:
            .double 100.0, 110.0, 120.0, 125.0, 140.0, 150.0, 155.0, 160.0, 165.0, 175.0, 180.0, 190.0, 200.0, 210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 180.0, 200.0, 220.0, 240.0, 260.0, 280.0, 300.0, 210.0, 230.0, 250.0, 270.0, 290.0, 310.0

        outputs:
            .double 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0

        size:
            .double 32.0
        size_int:
            .quad 32
        learning_rate:
            .double 0.01
        epochs:
            .quad 1000

        input_height:
            .double 79.0
        input_weight:
            .double 250.0


.section .text
    fmt:
        .string "%d\n"

.global main
.extern printf
.extern exp


main:
    push %rbp
    mov %rsp, %rbp

    call train_logistic_2d

    movsd %xmm2, %xmm3              # Move b into xmm3 for predict_class_2d
    movsd %xmm1, %xmm2              # Move w2 into xmm2 for predict_class_2d
    movsd %xmm0, %xmm1              # Move w1 into xmm1 for predict_class_2d

    movsd input_height(%rip), %xmm4 # Load input_height into xmm4 for predict_class_2d
    movsd input_weight(%rip), %xmm5 # Load input_weight into xmm5 for predict_class_2d

    call predict_class_2d           # Get predicted class in eax

    movzbl %al, %eax                # Zero-extend predicted class to eax for printf
    mov %eax, %esi                  # Move predicted class into esi for printf
    lea fmt(%rip), %rdi             # Load format string into rdi for printf
    xor %rax, %rax                  # Clear rax for variadic function call
    call printf                     # Print predicted class

    mov $0, %eax                    # Return 0 from main
    pop %rbp
    ret


/*
* Sigmoid function for logistic regression
* inputs:
*        xmm0: z (the linear combination of inputs and weights)
* returns:
*        xmm0: sigmoid(z)
*/
sigmoid:

    movsd negone(%rip), %xmm1        # Load -1.0 into xmm2

    mulsd %xmm1, %xmm0               # Negate z for exp(-z)
    call exp                         # Compute exp(-z)

    movsd one(%rip), %xmm1           # Load 1.0 into xmm2
 
    addsd %xmm1, %xmm0               # Compute 1 + exp(-z)
    divsd %xmm0, %xmm1               # Compute 1 / (1 + exp(-z))
    movsd %xmm1, %xmm0               # Move result to xmm0 for return

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
 = w1 * heights[i]
        mulsd heights(,%rsi,8), %xmm12

        movsd %xmm1, %xmm9          # temp = w2 * weights[i]
        mulsd weights(,%rsi,8), %xmm9

        addsd %xmm9, %xmm12         # z += temp
        addsd %xmm2, %xmm12         # z += b

        subq $16, %rsp              # Push xmm0 (w1) before sigmoid 
        movups %xmm0, (%rsp)
        subq $16, %rsp              # Push xmm1 (w2) before sigmoid
        movups %xmm1, (%rsp)
        subq $16, %rsp              # Push xmm2 (b) before sigmoid
        movups %xmm2, (%rsp)

        push %rsi                   # Save loop counter for sigmoid
        push %rdi                   # Save epochs for sigmoid
        
        movsd %xmm12, %xmm0         # Move z into xmm0 for sigmoid

        call sigmoid                # p = sigmoid(z)        (p to be held in xmm12)
        movsd %xmm0 , %xmm12

        pop %rdi                    # Restore epochs
        pop %rsi                    # Restore loop counter

        movups (%rsp), %xmm2        # Restore b
        addq $16, %rsp
        movups (%rsp), %xmm1        # Restore w2
        addq $16, %rsp
        movups (%rsp), %xmm0        # Restore w1
        addq $16, %rsp
      
        subsd outputs(,%rsi,8), %xmm12 # error = p - y[i]   (error to be held in xmm12)

        movsd %xmm12, %xmm9         # temp1 = error
        mulsd heights(,%rsi,8), %xmm9 # temp1 *= height[i]
        addsd %xmm9, %xmm13         # dw1 += temp1

        movsd %xmm12, %xmm10        # temp2 = error
        mulsd weights(,%rsi,8), %xmm10 # temp2 *= weight[i]
        addsd %xmm10, %xmm14        # dw2 += temp2

        addsd %xmm12, %xmm15        # db += error

        inc %rsi                    # i++
        cmp %rdi, %rsi              # Compare i with size
        jl loop2                    # Loop if i < size
    # End loop2
    
      pop %rdi                       # Restore epochs
      pop %rsi                       # Restore loop counter

      divsd size(%rip), %xmm13       # dw1 /= size
      divsd size(%rip), %xmm14       # dw2 /= size
      divsd size(%rip), %xmm15       # db /= size

      movsd learning_rate(%rip), %xmm9 # temp1 = learning_rate
      mulsd %xmm13, %xmm9            # temp1 *= dw1
      subsd %xmm9, %xmm0             # w1 -= temp1

      movsd learning_rate(%rip), %xmm9 # temp1 = learning_rate
      mulsd %xmm14, %xmm9            # temp1 *= dw2
      subsd %xmm9, %xmm1             # w2 -= temp1

      movsd learning_rate(%rip), %xmm9 # temp1 = learning_rate
      mulsd %xmm15, %xmm9            # temp1 *= db
      subsd %xmm9, %xmm2             # b -= temp1

      inc %rsi                       # epoch++
      cmp %rdi, %rsi                 # Compare epoch with epochs
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

    movsd %xmm1, %xmm0               # Move w1 into xmm0 for computation
    mulsd %xmm4, %xmm0              # w1 * input_height

    movsd %xmm2, %xmm6              # Move w2 into xmm6
    mulsd %xmm5, %xmm6              # w2 * input_weight

    addsd %xmm6, %xmm0              # w1 * input_height + w2 * input_weight
    addsd %xmm3, %xmm0              # + b

    # expf inside sigmoid might consume some of these, i dunno they're here to be safe for the time being
    subq $16, %rsp                  # Push xmm0 (z) before sigmoid
    movups %xmm0, (%rsp)
    subq $16, %rsp                  # Push xmm1 (w1) before sigmoid 
    movups %xmm1, (%rsp)
    subq $16, %rsp                  # Push xmm2 (w2) before sigmoid 
    movups %xmm2, (%rsp)
    subq $16, %rsp                  # Push xmm3 (b) before sigmoid 
    movups %xmm3, (%rsp)
    subq $16, %rsp                  # Push xmm4 (input_height) before sigmoid
    movups %xmm4, (%rsp)
    subq $16, %rsp                  # Push xmm5 (input_weight) before sigmoid
    movups %xmm5, (%rsp)

    call sigmoid                    # predicted probability = sigmoid(z)

    movups (%rsp), %xmm5            # Restore input_weight
    addq $16, %rsp
    movups (%rsp), %xmm4            # Restore input_height
    addq $16, %rsp
    movups (%rsp), %xmm3            # Restore b
    addq $16, %rsp
    movups (%rsp), %xmm2            # Restore w2
    addq $16, %rsp
    movups (%rsp), %xmm1            # Restore w1
    addq $16, %rsp
    movups (%rsp), %xmm0            # Restore z (though it's not needed since sigmoid already put the result in xmm0)
    addq $16, %rsp

    ret


/**
* Predicts class for a 2D logistic regression model based on a threshold of 0.5
* inputs:
*        xmm1: w1
*        xmm2: w2
*        xmm3: b
*        xmm4: input_height
*        xmm5: input_weight
* returns:
*        xmm0: predicted class (0 or 1)
*/ 
predict_class_2d:

    call predict_prob_2d            # Get predicted probability in xmm0

    subq $16, %rsp                  # Push xmm1 (w1)
    movups %xmm1, (%rsp)
    subq $16, %rsp                  # Push xmm2 (w2)
    movups %xmm2, (%rsp)

    movsd %xmm0, %xmm1              # Move predicted probability into xmm1 for comparison
    movsd prob_threshold(%rip), %xmm2 # Load 0.5 into xmm2

    comisd %xmm1, %xmm2             # Compare predicted probability with 0.5
    setb %al                        # Set al to 1 if predicted probability < 0.5, else set to 0
    movzbl %al, %eax                # Zero-extend al to eax for return value

    movups (%rsp), %xmm2            # Restore w2
    addq $16, %rsp
    movups (%rsp), %xmm1            # Restore w1
    addq $16, %rsp

    ret

.section .note.GNU-stack,"",@progbits
