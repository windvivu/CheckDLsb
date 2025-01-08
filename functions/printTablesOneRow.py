from rich import console

console = console.Console()

def print2TableOneRow(table1, table2, space = ' '):
     # Tính toán độ rộng của bảng
    # width = max(len(line) for table in [table1, table2] for line in table.split("\n"))
    try:
        for row1, row2 in zip(table1.split("\n"), table2.split("\n")):
            console.print(row1 + space + row2)
    except:
        pass

def print3TableOneRow(table1, table2, table3, space = ' '):
    # In ba bảng cạnh nhau
    try:
        for row1, row2, row3 in zip(table1.split("\n"), table2.split("\n"), table3.split("\n")):
            console.print(row1 + space + row2 + space + row3)
    except:
        pass

def print4TableOneRow(table1, table2, table3, table4, space = ' '):
    # In bốn bảng cạnh nhau
    try:
        for row1, row2, row3, row4 in zip(table1.split("\n"), table2.split("\n"), table3.split("\n"), table4.split("\n")):
            console.print(row1 + space + row2 + space + row3 + space + row4)
    except:
        pass

