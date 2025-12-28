.model small
.stack 100h
print_decimal proc
	MOV AX,DX
	MOV CX,0
	MOV BX,10

	again:
		MOV DX,0
		DIV BX
		PUSH DX
		INC CX
		CMP AX,0
		JNE again
	
	again1:
		POP DX
		ADD DX,30H
		MOV AH,02
		INT 21H
		DEC CX
		JNZ again1

	RET
print_decimal endp