#!/usr/bin/env python3
# Lua Decompiler - Converted from VB to Python
# Original VB code by 4E534B

import struct
import os
import datetime
from enum import IntEnum, auto
from typing import List, Dict, Tuple, Optional, Any, Union

# Constants from the original VB code
LUA_BinaryID = 0x1B           # binary files start with ESC (27 or 0x1B)
LUA_Signature = b"Lua"        # signature following the ESC
LUA_Version = 0x40            # 64 or "@"
LUA_Endianess = 1             # Must have one in the endianess

# Test size values for LUA virtual machine (in bytes)
LUA_TS_int = 4
LUA_TS_size_t = 4
LUA_TS_Instruction = 4

# In bits
LUA_TS_SIZE_INSTRUCTION = 32
LUA_TS_SIZE_SIZE_OP = 6
LUA_TS_SIZE_B = 9

# In bytes again
LUA_TS_Number = 8            # A double

LUA_TestNumber = 314159265.358979  # a multiple of PI for testing native format

# All strings that denote a token. Separated by spaces.
LUA_Tokens = "~= <= >= < > == = + - * / % ( ) { } [ ] ; , . .. ..."

# Masks for instruction decoding
LUA_Instruction_Mask = 63           # ie [bits:00011111 00000000 00000000 00000000]
LUA_BArg_Mask = 511                 # ie [bits:11111111 00000001 00000000 00000000]
LUA_Intruction_SArg_Zero = 33554431 # ie 2^25 - 1 for -33554431 to 33554431 (which is ~2^26 values)

# Constants for instruction parsing
LUA_Size_OP = 6
LUA_Size_A = 17
LUA_Size_B = 9

LUA_Pos_U = LUA_Size_OP
LUA_Pos_S = LUA_Size_OP
LUA_Pos_A = LUA_Size_OP + LUA_Size_B
LUA_Pos_B = LUA_Size_OP

LUA_MaxStackSize = 1024        # Max. (internal) stack size.
LUA_NumJumpsStored = 128       # NR. of jumps stored (may be nested)
LUA_FieldsPerFlush = 64        # Fields per flush. Looks like max. stack size...
LUA_ExtraFields = 8            # Extra fields for calculating the stack size

LUA_Null = "nil"               # nil value in Lua

# Function statements and their search strings
FuncStatement_SStr = "--<< Position Reserved for Function "
FuncStatement_SStr_R = " >>--"
FuncStatement = f"{FuncStatement_SStr}%n, &f, &p{FuncStatement_SStr_R}"

# OPCodes enum (from modLUAInstruction.bas)
class LUA_OPCodes(IntEnum):
    OP_END = 0
    OP_RETURN = 1
    OP_CALL = 2
    OP_TAILCALL = 3
    OP_PUSHNIL = 4
    OP_POP = 5
    OP_PUSHINT = 6
    OP_PUSHSTRING = 7
    OP_PUSHNUM = 8
    OP_PUSHNEGNUM = 9
    OP_PUSHUPVALUE = 10
    OP_GETLOCAL = 11
    OP_GETGLOBAL = 12
    OP_GETTABLE = 13
    OP_GETDOTTED = 14
    OP_GETINDEXED = 15
    OP_PUSHSELF = 16
    OP_CREATETABLE = 17
    OP_SETLOCAL = 18
    OP_SETGLOBAL = 19
    OP_SETTABLE = 20
    OP_SETLIST = 21
    OP_SETMAP = 22
    OP_ADD = 23
    OP_ADDI = 24
    OP_SUB = 25
    OP_MULT = 26
    OP_DIV = 27
    OP_POW = 28
    OP_CONCAT = 29
    OP_MINUS = 30
    OP_NOT = 31
    OP_JMPNE = 32
    OP_JMPEQ = 33
    OP_JMPLT = 34
    OP_JMPLE = 35
    OP_JMPGT = 36
    OP_JMPGE = 37
    OP_JMPT = 38
    OP_JMPF = 39
    OP_JMPONT = 40
    OP_JMPONF = 41
    OP_JMP = 42
    OP_PUSHNILJMP = 43
    OP_FORPREP = 44
    OP_FORLOOP = 45
    OP_LFORPREP = 46
    OP_LFORLOOP = 47
    OP_CLOSURE = 48

# Interpretation data types
class InterpretationDataType(IntEnum):
    IDT_Nil = 0
    IDT_Integral = 1
    IDT_Float = 2
    IDT_Char = 3
    IDT_Table = 4
    IDT_LocalVar = 5
    IDT_Closure = 6

# Interpretation data flags
class InterpretationDataFlags(IntEnum):
    IDF_IsALocalValue = 1
    IDF_FunctionReturn = 2
    IDF_FunctionReturnWithEQ = 4

# Data structures
class LUA_LocalVariable:
    def __init__(self):
        self.name = ""
        self.startpc = 0
        self.endpc = 0

class LUA_String:
    def __init__(self):
        self.data = ""

class LUA_Chunk:
    def __init__(self):
        self.source = ""
        self.lineDefined = 0
        self.numParams = 0
        self.isVarArg = False
        self.maxStackSize = 0
        
        self.locals = []
        self.lineInfo = []
        
        self.strings = []
        self.numbers = []
        self.functions = []
        
        self.instructions = []
        
        self.funcPtr = 0
        self.parentPtr = 0

class LUA_Header:
    def __init__(self):
        self.binID = 0
        self.sign = ""
        self.vers = 0
        self.byteOrder = 0
        
        self.ts_int = 0
        self.ts_size_t = 0
        self.ts_instruction = 0
        
        self.ts_size_instruction = 0
        self.ts_size_size_op = 0
        self.ts_size_b = 0
        
        self.ts_number = 0
        
        self.testNumber = 0.0

class LUA_File:
    def __init__(self):
        self.hdr = LUA_Header()
        self.chunks = []
        self.funcs = [LUA_Chunk() for _ in range(MAX_FUNCTIONS)]

class InterpretationStack:
    def __init__(self):
        self.value = ""
        self.type = InterpretationDataType.IDT_Nil
        self.flags = 0
        self.extraValue = 0
        self.extraString = ""

# Global variables
MAX_FUNCTIONS = 256
localProcessed = []
stack = []
stackP = 0
currIns = 0
level = 0

# Utility functions
def shift_right(value, bits):
    """Implement VB's bitwise shift right"""
    return value >> bits

def chop_terminating_null(s):
    """Remove terminating null character if present"""
    if s and s[-1] == '\0':
        return s[:-1]
    return s

def max_val(a, b):
    """Return the maximum of two values"""
    return max(a, b)

def min_val(a, b):
    """Return the minimum of two values"""
    return min(a, b)

def nearest_power_of_base(value, base):
    """Find the nearest power of base >= value"""
    result = 1
    while result < value:
        result *= base
    return result

def reverse_condition(cond_str):
    """Reverse a condition string"""
    if cond_str == "==": return "~="
    if cond_str == "~=": return "=="
    if cond_str == "<": return ">="
    if cond_str == ">": return "<="
    if cond_str == "<=": return ">"
    if cond_str == ">=": return "<"
    return cond_str

def unquote_string(s):
    """Remove quotes from a string if present"""
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    if s.startswith('[[') and s.endswith(']]'):
        return s[2:-2]
    return s

def kill_spaces(s):
    """Remove excessive spaces"""
    while "  " in s:
        s = s.replace("  ", " ")
    return s.strip()

def bits_present(value, flags):
    """Check if bits are present in value"""
    return (value & flags) == flags

def escape_backslashes(s):
    """Escape backslashes in a string"""
    return s.replace("\\", "\\\\")

def instruction_get_opcode(instruction):
    """Extract the opcode from an instruction"""
    return instruction & LUA_Instruction_Mask

def instruction_get_uarg(instruction):
    """Extract the U (unsigned) argument from an instruction"""
    return shift_right(instruction, LUA_Pos_U)

def instruction_get_sarg(instruction):
    """Extract the S (signed) argument from an instruction"""
    return shift_right(instruction, LUA_Pos_S) - LUA_Intruction_SArg_Zero

def instruction_get_aarg(instruction):
    """Extract the A argument from an instruction"""
    return shift_right(instruction, LUA_Pos_A)

def instruction_get_barg(instruction):
    """Extract the B argument from an instruction"""
    return shift_right(instruction, LUA_Pos_B) & LUA_BArg_Mask

def reverse_condition_op(condition):
    """Reverse the condition opcode"""
    if condition == LUA_OPCodes.OP_JMPEQ:
        return LUA_OPCodes.OP_JMPNE
    elif condition == LUA_OPCodes.OP_JMPNE:
        return LUA_OPCodes.OP_JMPEQ
    elif condition == LUA_OPCodes.OP_JMPLT:
        return LUA_OPCodes.OP_JMPGE
    elif condition == LUA_OPCodes.OP_JMPGT:
        return LUA_OPCodes.OP_JMPLE
    elif condition == LUA_OPCodes.OP_JMPLE:
        return LUA_OPCodes.OP_JMPGT
    elif condition == LUA_OPCodes.OP_JMPGE:
        return LUA_OPCodes.OP_JMPLT
    return condition

# Main functionality
def read_lua_function(file, out_func, parent_chunk, the_lua):
    """Read a Lua function from the file"""
    global func_call_nr
    
    func_call_nr += 1
    
    # Initialize the function structure
    out_func.locals = []
    out_func.lineInfo = []
    out_func.strings = []
    out_func.numbers = []
    out_func.instructions = []
    
    # Read chunk header
    len_source = struct.unpack("<I", file.read(4))[0]
    out_func.source = file.read(len_source).decode('latin-1')
    out_func.source = chop_terminating_null(out_func.source)
    
    out_func.lineDefined = struct.unpack("<I", file.read(4))[0]
    out_func.numParams = struct.unpack("<I", file.read(4))[0]
    out_func.isVarArg = struct.unpack("<B", file.read(1))[0] != 0
    out_func.maxStackSize = struct.unpack("<I", file.read(4))[0]
    
    # Read local variables
    num_locals = struct.unpack("<I", file.read(4))[0]
    out_func.locals = []
    
    for i in range(num_locals):
        local = LUA_LocalVariable()
        name_len = struct.unpack("<I", file.read(4))[0]
        local.name = file.read(name_len).decode('latin-1')
        local.name = chop_terminating_null(local.name)
        
        local.startpc = struct.unpack("<I", file.read(4))[0]
        local.endpc = struct.unpack("<I", file.read(4))[0]
        out_func.locals.append(local)
    
    # Read line info
    num_line_info = struct.unpack("<I", file.read(4))[0]
    out_func.lineInfo = []
    
    if num_line_info > 0:
        for _ in range(num_line_info):
            out_func.lineInfo.append(struct.unpack("<I", file.read(4))[0])
    
    # Read constants - strings
    num_strings = struct.unpack("<I", file.read(4))[0]
    out_func.strings = []
    
    for _ in range(num_strings):
        lua_str = LUA_String()
        str_len = struct.unpack("<I", file.read(4))[0]
        lua_str.data = file.read(str_len).decode('latin-1')
        lua_str.data = chop_terminating_null(lua_str.data)
        out_func.strings.append(lua_str)
    
    # Read constants - numbers
    num_numbers = struct.unpack("<I", file.read(4))[0]
    out_func.numbers = []
    
    if num_numbers > 0:
        for _ in range(num_numbers):
            out_func.numbers.append(struct.unpack("<d", file.read(8))[0])
    
    # Read functions
    num_functions = struct.unpack("<I", file.read(4))[0]
    out_func.functions = []
    
    for i in range(num_functions):
        func_index = func_call_nr
        
        # Set pointers
        the_lua.funcs[func_index].funcPtr = id(the_lua.funcs[func_index])
        the_lua.funcs[func_index].parentPtr = out_func.funcPtr
        
        # Note the function index
        out_func.functions.append(the_lua.funcs[func_index].funcPtr)
        
        if the_lua.funcs[func_index].parentPtr == the_lua.funcs[func_index].funcPtr:
            the_lua.funcs[func_index].parentPtr = 0
        
        # Read the function
        read_lua_function(file, the_lua.funcs[func_index], out_func, the_lua)
    
    # Read instructions
    num_instructions = struct.unpack("<I", file.read(4))[0]
    out_func.instructions = []
    
    if num_instructions > 0:
        for _ in range(num_instructions):
            out_func.instructions.append(struct.unpack("<I", file.read(4))[0])
    
    return True

def read_lua(filename, out_lua):
    """Open and read a Lua binary file"""
    try:
        with open(filename, 'rb') as file:
            # Read header
            header_bytes = file.read(19)  # Size of LUA_Header in bytes
            
            if len(header_bytes) < 19:
                print("File size too small")
                return False
            
            out_lua.hdr.binID = header_bytes[0]
            out_lua.hdr.sign = header_bytes[1:4].decode('latin-1')
            out_lua.hdr.vers = header_bytes[4]
            out_lua.hdr.byteOrder = header_bytes[5]
            
            out_lua.hdr.ts_int = header_bytes[6]
            out_lua.hdr.ts_size_t = header_bytes[7]
            out_lua.hdr.ts_instruction = header_bytes[8]
            
            out_lua.hdr.ts_size_instruction = header_bytes[9]
            out_lua.hdr.ts_size_size_op = header_bytes[10]
            out_lua.hdr.ts_size_b = header_bytes[11]
            
            out_lua.hdr.ts_number = header_bytes[12]
            
            out_lua.hdr.testNumber = struct.unpack("<d", file.read(8))[0]
            
            # Verify Lua format
            test = (
                out_lua.hdr.binID == LUA_BinaryID and
                out_lua.hdr.sign == "Lua" and
                out_lua.hdr.vers == LUA_Version and
                out_lua.hdr.byteOrder == LUA_Endianess and
                out_lua.hdr.ts_int == LUA_TS_int and
                out_lua.hdr.ts_size_t == LUA_TS_size_t and
                out_lua.hdr.ts_instruction == LUA_TS_Instruction and
                out_lua.hdr.ts_size_instruction == LUA_TS_SIZE_INSTRUCTION and
                out_lua.hdr.ts_size_size_op == LUA_TS_SIZE_SIZE_OP and
                out_lua.hdr.ts_size_b == LUA_TS_SIZE_B and
                out_lua.hdr.ts_number == LUA_TS_Number and
                int(out_lua.hdr.testNumber) == int(LUA_TestNumber)
            )
            
            # Check file size
            file.seek(0, os.SEEK_END)
            if file.tell() == 0:
                print("Size of given file = 0")
                return False
            
            file.seek(19 + 8, os.SEEK_SET)  # Reset to after header + test number
            
            # Verify binary format
            if not (out_lua.hdr.binID == LUA_BinaryID and out_lua.hdr.sign == "Lua"):
                print("Need compiled binary files to be decompiled!")
                print("This does not seem to be a compiled LUA binary!")
                return False
            
            if not test:
                print("This LUA is probably corrupted. Please get this LUA recompiled through LuaC to ensure that the header is OK.")
                return False
            
            # Read all chunks
            out_lua.chunks = []
            
            while file.tell() < os.path.getsize(filename):
                chunk = LUA_Chunk()
                
                # Set pointers
                chunk.funcPtr = id(chunk)
                chunk.parentPtr = 0  # No parent
                
                out_lua.chunks.append(chunk)
                read_lua_function(file, chunk, chunk, out_lua)
            
            return True
    except Exception as e:
        print(f"Error reading Lua file: {e}")
        return False

def find_matching_start_pc_for_local(lua_chunk, find):
    """Find the index for the FIRST local with matching instruction number (startpc)"""
    for i, local in enumerate(lua_chunk.locals):
        if local.startpc == find:
            return i
    return 0

def find_matching_start_pc_for_last_local(lua_chunk, find):
    """Find the index for the LAST local with matching instruction number (startpc)"""
    last_index = -1
    for i, local in enumerate(lua_chunk.locals):
        if local.startpc == find:
            last_index = i
    return last_index

def find_num_locals_with_end_pc(lua_chunk, find):
    """Return the number of locals which end at the given instruction"""
    count = 0
    for local in lua_chunk.locals:
        if local.endpc == find:
            count += 1
    return count

def find_num_locals_with_start_pc(lua_chunk, find):
    """Return the number of locals which start at the given instruction"""
    count = 0
    for local in lua_chunk.locals:
        if local.startpc == find:
            count += 1
    return count

def find_line_info(lua_chunk, ins):
    """Find the line info for given instruction number"""
    # Not implemented for brevity as it's complex
    return -1

def process_value_for_output(offset=0):
    """Process a value for output"""
    global stackP, stack
    
    if stack[stackP + offset].type == InterpretationDataType.IDT_Table:
        if stack[stackP + offset].value.isdigit():
            # Empty table
            if int(stack[stackP + offset].value) == 0:
                return "{}"
            else:
                # Data, but not defined yet...
                result = "{"
                for i in range(1, int(stack[stackP + offset].value) + 1):
                    result += LUA_Null
                    if i != int(stack[stackP + offset].value):
                        result += ", "
                result += "}"
                return result
        else:
            # Filled table
            result = stack[stackP + offset].value
            return escape_backslashes(result)
    elif stack[stackP + offset].type == InterpretationDataType.IDT_Nil:
        return LUA_Null
    else:
        result = (
            " " if "[[" in stack[stackP + offset].value else ""
        ) + stack[stackP + offset].value
        
        return escape_backslashes(result)

def process_table_value(tbl, key):
    """Process a table value with the key"""
    dotted = False  # Simplified from original
    
    if dotted:
        return f"{tbl}.{key}"
    else:
        return f"{tbl}[{key}]"

def push_statement(out, statement, semi_colon=True, crlf=True):
    """Process a statement for output"""
    global level
    
    tmp_str = statement
    
    # Remove multiple spaces
    while "  " in tmp_str:
        tmp_str = tmp_str.replace("  ", " ")
    
    # Dump statement
    out += (
        "\t" * max(0, level) +
        tmp_str.strip() +
        ("" if statement.startswith("--") else (";" if semi_colon and statement else "")) +
        ("\n" if crlf else "")
    )
    
    return out

def push_value_in_stack(value, dtype, instruction_nr, lua_chunk, out_lua, search_for_local=True):
    """Push a value into the stack"""
    global stackP, stack, localProcessed
    
    if search_for_local:
        # Try to search for a feasible local with 'active' state
        for j in range(find_matching_start_pc_for_local(lua_chunk, instruction_nr), 
                       find_matching_start_pc_for_last_local(lua_chunk, instruction_nr) + 1):
            if j >= 0 and not localProcessed[j]:
                pop_local_from_stack(lua_chunk.locals[j].name)
                i = push_local_in_stack(value, dtype)
                
                stack[i].extraString = lua_chunk.locals[j].name
                stack[i].flags = InterpretationDataFlags.IDF_IsALocalValue
                
                tmp_str = f"local {lua_chunk.locals[j].name}={process_value_for_output()}"
                
                # Mark this local done
                localProcessed[j] = True
                
                # Add to output
                out_lua = push_statement(out_lua, tmp_str)
                
                return out_lua
    
    # If no local was found or search_for_local is False
    stackP += 1
    
    # Ensure we have enough stack space
    while len(stack) <= stackP:
        stack.append(InterpretationStack())
    
    stack[stackP].value = value
    stack[stackP].type = dtype
    
    return out_lua

def push_value_in_stack_top(value, dtype):
    """Push a value into the stack at the top"""
    global stackP, stack
    
    stackP += 1
    
    # Ensure we have enough stack space
    while len(stack) <= stackP:
        stack.append(InterpretationStack())
    
    for i in range(stackP - 1, 0, -1):
        if not (stack[i].type == InterpretationDataType.IDT_LocalVar or 
                bits_present(stack[i].flags, InterpretationDataFlags.IDF_IsALocalValue)):
            stack[i + 1].type = stack[i].type
            stack[i + 1].value = stack[i].value
            stack[i + 1].flags = stack[i].flags
            stack[i + 1].extraString = stack[i].extraString
            stack[i + 1].extraValue = stack[i].extraValue
        else:
            # Don't move a local var
            stack[i + 1].type = dtype
            stack[i + 1].value = value
            stack[i + 1].flags = 0
            stack[i + 1].extraString = ""
            stack[i + 1].extraValue = 0
            
            return i + 1
    
    stack[1].type = dtype
    stack[1].value = value
    stack[1].flags = 0
    stack[1].extraString = ""
    stack[1].extraValue = 0
    
    return 1

def push_local_in_stack(value, dtype):
    """Push a local in the stack"""
    result = push_value_in_stack_top(value, dtype)
    stack[result].flags = InterpretationDataFlags.IDF_IsALocalValue
    return result

def pop_value_from_stack(num_values=1):
    """Pop values from the stack"""
    global stackP, stack
    
    for _ in range(min(num_values, stackP)):
        stack[stackP].value = ""
        stack[stackP].type = InterpretationDataType.IDT_Nil
        stack[stackP].flags = 0
        stack[stackP].extraValue = 0
        stack[stackP].extraString = ""
        
        stackP -= 1

def pop_value_from_stack_top(num_values=1):
    """Pop values from the stack top"""
    global stackP, stack
    
    for i in range(num_values + 1, stackP + 1):
        stack[i - num_values].type = stack[i].type
        stack[i - num_values].value = stack[i].value
        stack[i - num_values].flags = stack[i].flags
        stack[i - num_values].extraString = stack[i].extraString
        stack[i - num_values].extraValue = stack[i].extraValue
    
    pop_value_from_stack(num_values)

def pop_local_from_stack(name=""):
    """Pop a local from the stack"""
    global stackP, stack
    
    for i in range(1, stackP + 1):
        move = False
        
        if name == "":
            if i == stackP:
                move = True
                break
            
            if not (stack[i + 1].type == InterpretationDataType.IDT_LocalVar or 
                    bits_present(stack[i + 1].flags, InterpretationDataFlags.IDF_IsALocalValue)):
                move = True
        else:
            if ((stack[i].value == name and stack[i].type == InterpretationDataType.IDT_LocalVar) or 
                (stack[i].extraString == name and 
                 bits_present(stack[i].flags, InterpretationDataFlags.IDF_IsALocalValue))):
                move = True
        
        if move:
            if i == stackP:
                break
            
            stack[i].type = stack[i + 1].type
            stack[i].value = stack[i + 1].value
            stack[i].flags = stack[i + 1].flags
            stack[i].extraString = stack[i + 1].extraString
            stack[i].extraValue = stack[i + 1].extraValue
    
    # Finally pop a value using default method
    if move or i == 1:
        pop_value_from_stack()

def init(lua_chunk, tab_level):
    """Initialize the decompiler state"""
    global stackP, level, currIns, stack, localProcessed
    
    stackP = 0
    level = tab_level
    
    if len(lua_chunk.instructions) > 0:
        currIns = lua_chunk.instructions[0]
    else:
        currIns = 0
    
    stack = [InterpretationStack() for _ in range(min(
        nearest_power_of_base(lua_chunk.maxStackSize + LUA_ExtraFields, 2),
        LUA_MaxStackSize
    ) + 1)]
    
    if len(lua_chunk.locals) > 0:
        localProcessed = [False] * (len(lua_chunk.locals) + 1)
    else:
        localProcessed = [False]
    
    return True

def process_locals(lua_chunk, i):
    """Process locals at instruction i"""
    global localProcessed
    
    # Static variable in original VB code
    global last_local
    if i == 0:
        last_local = 0
    
    # Initialize each local
    for j in range(last_local, len(lua_chunk.locals)):
        if lua_chunk.locals[j].startpc == i:
            push_local_in_stack(lua_chunk.locals[j].name, InterpretationDataType.IDT_LocalVar)
            last_local = j + 1
        
        if lua_chunk.locals[j].startpc > i:
            break

def process_do_end_chunks(lua_chunk, str_for_do, str_for_end):
    """Process DO and END chunks"""
    for i in range(len(lua_chunk.locals) - 1):
        # Whether this does not end at the last instruction
        cond_a = not (lua_chunk.locals[i].endpc == len(lua_chunk.instructions) - 1)
        
        # Whether this 'do...end' chunk hasn't been added yet
        cond_b = not (lua_chunk.locals[i].endpc == tmp_long2)
        
        # Whether this is not a local var of 'for'
        if lua_chunk.locals[i].startpc > 0:
            cond_c = (not (instruction_get_opcode(lua_chunk.instructions[lua_chunk.locals[i].endpc]) == LUA_OPCodes.OP_POP) and
                     not (LUA_OPCodes.OP_FORPREP <= instruction_get_opcode(lua_chunk.instructions[lua_chunk.locals[i].startpc]) <= LUA_OPCodes.OP_LFORLOOP))
        else:
            cond_c = False  # startPC = 0; of a function arg; 'do...end' not needed
        
        if cond_a and cond_b and cond_c:
            tmp_long = lua_chunk.locals[i].startpc
            tmp_long2 = lua_chunk.locals[i].endpc
            
            # Add DO chunk
            if tmp_long > 0 and tmp_long2 > 0:
                str_for_do = f",{tmp_long}{str_for_do}"
                str_for_end = f",{tmp_long2}{str_for_end}"
    
    return str_for_do, str_for_end

def process_do_chunk(lua_chunk, i, str_for_do, out_lua):
    """Process DO chunk at instruction i"""
    global level
    
    tmp_str = ""
    
    starting_label = True
    while starting_label:
        starting_label = False
        tmp_long = str_for_do.rfind(",")
        if tmp_long < len(str_for_do) - 1:
            tmp_str = str_for_do[tmp_long + 1:]
        
        if tmp_str and tmp_long > 0:
            # Is this the one?
            if int(tmp_str) == i:
                out_lua = push_statement(out_lua, "do", False)
                level += 1
                
                # Chop off this part
                str_for_do = str_for_do[:tmp_long]
                
                starting_label = True  # Because we need to work on all
    
    return out_lua, str_for_do

def process_end_statement(lua_chunk, i, str_for_end, str_for_jump_type, out_lua):
    """Process END statement at instruction i"""
    global level
    
    tmp_str = ""
    
    starting_label = True
    while starting_label:
        starting_label = False
        tmp_long = str_for_end.rfind(",")
        
        if tmp_long < len(str_for_end) - 1:
            tmp_str = str_for_end[tmp_long + 1:]
        
        if tmp_str and tmp_long > 0:
            # Is this the one?
            if int(tmp_str) == i:
                level -= 1
                out_lua = push_statement(out_lua, "end", False)
                out_lua = push_statement(out_lua, "")
                
                # Chop off this part
                str_for_end = str_for_end[:tmp_long]
                str_for_jump_type = str_for_jump_type[:-1]
                
                starting_label = True
    
    return out_lua, str_for_end, str_for_jump_type

def find_next_condition(curr_ins, ins, lua_chunk, include_jmp=True, override_line_info=False):
    """Find the next condition for the 'if' line"""
    # Try to get the next nil-comparison jump if possible
    if len(lua_chunk.instructions) > ins + 2:
        op_code = instruction_get_opcode(lua_chunk.instructions[ins + 2])
        
        # Determine whether the next instruction is a jump
        if (LUA_OPCodes.OP_JMPNE <= op_code <= 
            (LUA_OPCodes.OP_JMP if include_jmp else LUA_OPCodes.OP_JMPONF)):
            if find_line_info(lua_chunk, ins + 2) == find_line_info(lua_chunk, ins) or override_line_info:
                return ins + 2
    
    # Try to get the next two-comparison jump if possible
    if len(lua_chunk.instructions) > ins + 3:
        op_code = instruction_get_opcode(lua_chunk.instructions[ins + 3])
        
        # Determine whether the instruction is a jump
        if (LUA_OPCodes.OP_JMPNE <= op_code <= 
            (LUA_OPCodes.OP_JMP if include_jmp else LUA_OPCodes.OP_JMPONF)):
            if find_line_info(lua_chunk, ins + 3) == find_line_info(lua_chunk, ins) or override_line_info:
                return ins + 3
    
    op_code = instruction_get_opcode(lua_chunk.instructions[ins + instruction_get_sarg(curr_ins)])
    
    # Determine whether the instruction is a jump
    if (LUA_OPCodes.OP_JMPNE <= op_code <= 
        (LUA_OPCodes.OP_JMP if include_jmp else LUA_OPCodes.OP_JMPONF)):
        if (find_line_info(lua_chunk, ins + instruction_get_sarg(curr_ins)) == 
            find_line_info(lua_chunk, ins)) or override_line_info:
            return ins + instruction_get_sarg(curr_ins)
    
    return 0

def find_end_of_jump(curr_ins, ins, lua_chunk):
    """Find the end of this 'if' jump"""
    last_value = ins
    next_cond = find_next_condition(curr_ins, ins, lua_chunk)
    
    while next_cond != 0:
        last_value = next_cond
        next_cond = find_next_condition(lua_chunk.instructions[last_value], last_value, lua_chunk)
        
        # Safety catch: Jumps with 0 pc offset!
        if last_value == next_cond:
            break
    
    return instruction_get_sarg(lua_chunk.instructions[last_value]) + last_value + 1

def find_lua_function(func_ptr, the_lua):
    """Find the LUA function with this pointer"""
    for chunk in the_lua.chunks:
        if chunk.funcPtr == func_ptr:
            return chunk
    
    for i, func in enumerate(the_lua.funcs):
        if func.funcPtr == 0:
            break
        
        if func.funcPtr == func_ptr:
            return func
    
    # We don't 'know' this function
    if func_ptr != 0:
        assert False, "Unknown function pointer"
    
    return None

def find_lua_function_tab(func_ptr, the_lua):
    """Get the tab level for the function with the given pointer"""
    the_func = find_lua_function(func_ptr, the_lua)
    tab_count = 0
    
    while the_func and the_func.parentPtr != 0:
        # For each valid parent, we have an extra tab
        tab_count += 1
        
        # Get this function's parent
        if the_func.parentPtr != 0:
            the_func = find_lua_function(the_func.parentPtr, the_lua)
    
    return tab_count

def lua_decompile_chunk(lua_chunk, parent_chunk, out_lua, tab_level):
    """Decompile a Lua chunk/function"""
    global level, currIns, stackP
    
    init(lua_chunk, tab_level)
    
    str_for_condition = ""
    str_for_do = ""
    str_for_end = ""
    str_for_jump_type = ""
    
    str_for_do, str_for_end = process_do_end_chunks(lua_chunk, str_for_do, str_for_end)
    process_locals(lua_chunk, 0)  # Observed in functions
    
    for i in range(len(lua_chunk.instructions)):
        currIns = lua_chunk.instructions[i]  # Next instruction
        ins_opcode = instruction_get_opcode(currIns)
        
        process_locals(lua_chunk, i)
        out_lua, str_for_do = process_do_chunk(lua_chunk, i, str_for_do, out_lua)
        out_lua, str_for_end, str_for_jump_type = process_end_statement(lua_chunk, i, str_for_end, str_for_jump_type, out_lua)
        
        if ins_opcode == LUA_OPCodes.OP_END:
            # Pop all vars ending here
            for _ in range(find_num_locals_with_end_pc(lua_chunk, i)):
                pop_local_from_stack()
            
            if stackP != 0:
                print("Warning! Stack is not empty! The written LUA may have been incorrectly decompiled!")
                assert stackP == 0
            
            break
        
        elif ins_opcode == LUA_OPCodes.OP_RETURN:
            # If this is not the second last instruction, then put in another block
            tmp_long = instruction_get_uarg(currIns) + 1
            tmp_long2 = ((instruction_get_opcode(lua_chunk.instructions[i + 1]) == LUA_OPCodes.OP_JMP) or
                         (instruction_get_opcode(lua_chunk.instructions[i + 1]) == LUA_OPCodes.OP_END))
            
            tmp_str = ("" if tmp_long2 else "do ") + "return "
            
            for j in range(tmp_long, stackP + 1):
                if j != 0:
                    tmp_str += stack[j].value + ("," if j != stackP else " ")
            
            tmp_str += "" if tmp_long2 else "end"
            
            out_lua = push_statement(out_lua, tmp_str)
            out_lua = push_statement(out_lua, "")
            
            pop_value_from_stack(stackP - tmp_long + 1)
        
        elif ins_opcode in (LUA_OPCodes.OP_CALL, LUA_OPCodes.OP_TAILCALL):
            # Pop the locals, which will be awarded this function
            tmp_long = find_matching_start_pc_for_local(lua_chunk, i)
            for j in range(find_num_locals_with_start_pc(lua_chunk, i)):
                pop_local_from_stack(lua_chunk.locals[tmp_long + j].name)
            
            # Then get stack pos for function and num returns
            tmp_long = instruction_get_aarg(currIns) + 1
            tmp_long2 = (instruction_get_barg(currIns) if ins_opcode == LUA_OPCodes.OP_CALL else 0)
            
            tmp_str = stack[tmp_long].value + "("
            
            for j in range(tmp_long + 1, stackP + 1):
                tmp_str += stack[j].value + ("," if j != stackP else "")
            
            tmp_str += ")"
            
            # FUNCTION ASSIGNMENT TO INITIALIZED LOCAL: Different handling
            if find_matching_start_pc_for_local(lua_chunk, i) >= 0:
                # Pop function and its vars; and pump in this function (don't assign to local)
                pop_value_from_stack(stackP - tmp_long + 1)
                out_lua = push_value_in_stack(tmp_str, InterpretationDataType.IDT_Char, i, lua_chunk, out_lua, False)
                
                # Retrieve local names (all will be consecutive)
                tmp_str2 = ""
                for j in range(1, tmp_long2 + 1):
                    tmp_str2 += ("," if j != 1 else "") + lua_chunk.locals[find_matching_start_pc_for_local(lua_chunk, i) + j - 1].name
                
                out_lua = push_statement(out_lua, "local " + tmp_str2 + "=" + tmp_str)
                
                # Don't pop the function, since it's assigned to a local
                pop_value_from_stack(stackP - tmp_long)
                
                stack[stackP].extraString = kill_spaces(tmp_str2)
                stack[stackP].flags = InterpretationDataFlags.IDF_IsALocalValue
            else:
                # Pop function and its vars and pump in only one return... the function itself
                pop_value_from_stack(stackP - tmp_long + 1)
                out_lua = push_value_in_stack(tmp_str, InterpretationDataType.IDT_Char, i, lua_chunk, out_lua)
                
                stack[stackP].flags = InterpretationDataFlags.IDF_FunctionReturn
                stack[stackP].extraValue = tmp_long2
                
                # If no returns then print immediately
                if tmp_long2 == 0:
                    out_lua = push_statement(out_lua, stack[stackP].value)
                    pop_value_from_stack()
        
        elif ins_opcode == LUA_OPCodes.OP_PUSHNIL:
            for _ in range(instruction_get_uarg(currIns)):
                out_lua = push_value_in_stack(LUA_Null, InterpretationDataType.IDT_Nil, i, lua_chunk, out_lua)
        
        elif ins_opcode == LUA_OPCodes.OP_POP:
            pop_value_from_stack(instruction_get_uarg(currIns))
        
        elif ins_opcode in (LUA_OPCodes.OP_PUSHINT, LUA_OPCodes.OP_PUSHNUM, LUA_OPCodes.OP_PUSHNEGNUM, LUA_OPCodes.OP_PUSHSTRING):
            # Extract the appropriate thing and decide whether it's to be converted or not
            if ins_opcode == LUA_OPCodes.OP_PUSHINT:
                tmp_str = str(instruction_get_sarg(currIns))
            elif ins_opcode == LUA_OPCodes.OP_PUSHSTRING:
                tmp_str = lua_chunk.strings[instruction_get_uarg(currIns)].data
                tmp_long = ('"' in tmp_str) or ('\r' in tmp_str) or ('\n' in tmp_str)
                tmp_str = ("[[" if tmp_long else '"') + tmp_str + ("]]" if tmp_long else '"')
            else:
                # +ve or -ve number
                tmp_str = (str(lua_chunk.numbers[instruction_get_uarg(currIns)]) if ins_opcode == LUA_OPCodes.OP_PUSHNUM
                          else str(-lua_chunk.numbers[instruction_get_uarg(currIns)]))
            
            # Get the appropriate data-type for this
            tmp_idt = (InterpretationDataType.IDT_Integral if ins_opcode == LUA_OPCodes.OP_PUSHINT else
                      InterpretationDataType.IDT_Float if ins_opcode in (LUA_OPCodes.OP_PUSHNUM, LUA_OPCodes.OP_PUSHNEGNUM) else
                      InterpretationDataType.IDT_Char)
            
            # Push in stack
            out_lua = push_value_in_stack(tmp_str, tmp_idt, i, lua_chunk, out_lua)
        
        elif ins_opcode == LUA_OPCodes.OP_PUSHUPVALUE:
            # Check for valid parent chunk
            assert id(parent_chunk) != id(lua_chunk)
            
            if id(parent_chunk) != id(lua_chunk):
                tmp_str = "%" + parent_chunk.strings[instruction_get_uarg(currIns)].data
                out_lua = push_value_in_stack(tmp_str, InterpretationDataType.IDT_Char, i, lua_chunk, out_lua)
            else:
                print("Error: Trying to access an upvalue from a non-existent parent chunk. This value will be taken as 'nil'")
                out_lua = push_value_in_stack(LUA_Null, InterpretationDataType.IDT_Nil, i, lua_chunk, out_lua)
        
        elif ins_opcode in (LUA_OPCodes.OP_GETLOCAL, LUA_OPCodes.OP_GETGLOBAL):
            # Extract the appropriate name
            tmp_str = (lua_chunk.locals[instruction_get_uarg(currIns)].name if ins_opcode == LUA_OPCodes.OP_GETLOCAL
                      else lua_chunk.strings[instruction_get_uarg(currIns)].data)
            
            out_lua = push_value_in_stack(tmp_str, InterpretationDataType.IDT_Char, i, lua_chunk, out_lua)
        
        elif ins_opcode in (LUA_OPCodes.OP_GETTABLE, LUA_OPCodes.OP_GETDOTTED, LUA_OPCodes.OP_GETINDEXED):
            if ins_opcode != LUA_OPCodes.OP_GETTABLE:
                if ins_opcode == LUA_OPCodes.OP_GETDOTTED:
                    tmp_str = lua_chunk.strings[instruction_get_uarg(currIns)].data
                    tmp_str = '"' + tmp_str + '"'
                else:
                    tmp_str = lua_chunk.locals[instruction_get_uarg(currIns)].name
            else:
                tmp_str = process_value_for_output()
            
            # Extract the name and table entry
            tmp_str = process_table_value(stack[stackP - (0 if ins_opcode != LUA_OPCodes.OP_GETTABLE else 1)].value, tmp_str)
            
            # Don't forget to pop the value(s) first! Then push our result ("t[i]")
            pop_value_from_stack(1 + (1 if ins_opcode == LUA_OPCodes.OP_GETTABLE else 0))
            out_lua = push_value_in_stack(tmp_str, InterpretationDataType.IDT_Char, i, lua_chunk, out_lua)
            
        # ... and so on for other opcodes

def lua_decompile(in_lua_path, out_lua_path):
    """Main function for decompiling a LUA"""
    the_lua = LUA_File()
    
    if not read_lua(in_lua_path, the_lua):
        return False
    
    assert len(the_lua.chunks) >= 1
    
    out_lua = ""
    
    # First decompile chunk, then functions in it
    for i, chunk in enumerate(the_lua.chunks):
        chunk_output = ""
        if not lua_decompile_chunk(chunk, chunk, chunk_output, 0):
            return False
        
        out_lua += chunk_output
        
        # Process all functions
        for j, func in enumerate(the_lua.funcs):
            if func.funcPtr == 0:
                break
            
            func_output = ""
            if not lua_decompile_chunk(func, the_lua.chunks[i], func_output, find_lua_function_tab(func.funcPtr, the_lua)):
                return False
                
            # Insert function into output (replacing placeholder)
            # This is a simplified version of the original function replacement logic
            out_lua_lines = out_lua.split('\n')
            
            for k, line in enumerate(out_lua_lines):
                if FuncStatement_SStr in line:
                    tmp_str = line[line.find(FuncStatement_SStr) + len(FuncStatement_SStr):]
                    tmp_str = tmp_str.replace(FuncStatement_SStr, "")
                    tmp_str = tmp_str[:tmp_str.rfind(" ") - 1]
                    
                    tmp_str_arr = tmp_str.split(", ")
                    
                    tmp_long2 = find_lua_function_tab(func.funcPtr, the_lua)
                    tmp_long3 = int(tmp_str_arr[1])  # Pointer to function
                    tmp_long4 = int(tmp_str_arr[2])  # Pointer to function's parent
                    
                    if tmp_long3 == func.funcPtr:
                        replacement = "\t" * (tmp_long2 - 1) + "function " + kill_spaces(tmp_str_arr[0] if tmp_str_arr[0] != "%n" else "") + "("
                        
                        for param_idx in range(func.numParams):
                            replacement += func.locals[param_idx].name + ("," if param_idx != func.numParams - 1 else "")
                        
                        if func.isVarArg:
                            replacement += ("," if func.numParams > 0 else "") + "..."
                        
                        replacement += ")" + ("\n" + func_output + "\n" if func_output else " ")
                        replacement += "\t" * (tmp_long2 - 1) + "end"
                        
                        original = f"{FuncStatement_SStr}{tmp_str}{FuncStatement_SStr_R}"
                        out_lua_lines[k] = line.replace(original, replacement)
                        out_lua = "\n".join(out_lua_lines)
                        func_output = ""
                        break
    
    # Write output to file
    with open(out_lua_path, 'w') as f:
        # Add header
        current_time = datetime.datetime.now()
        f.write(f"-- Python Lua Decompiler v1.0\n")
        f.write(f"-- Based on Cold Fusion LUA Decompiler by 4E534B\n")
        f.write(f"-- Date: {current_time.strftime('%Y-%m-%d')} Time: {current_time.strftime('%H:%M:%S')}\n")
        f.write(f"-- On error(s), please report issues\n\n")
        
        # Write the decompiled Lua
        f.write(out_lua)
    
    return True

def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python lua_decompiler.py input.luac output.lua")
        return False
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"Decompiling {input_file} to {output_file}...")
    result = lua_decompile(input_file, output_file)
    
    if result:
        print("Decompilation completed successfully!")
    else:
        print("Decompilation failed!")
    
    return result

if __name__ == "__main__":
    func_call_nr = 0  # Initialize the static variable
    last_local = 0    # Initialize another static variable
    tmp_long2 = 0     # Another static variable used in process_do_end_chunks
    main()