import sqlite3
import random
import sys

if len(sys.argv) != 2:
    print("Invalid number of arguments!")
    exit(0)

__db_file = sys.argv[1]

# Подключаемся к базе данных
conn = sqlite3.connect(__db_file)
cursor = conn.cursor()

# Запрашиваем данные и перемешиваем их
cursor.execute("SELECT * FROM positions")
rows = cursor.fetchall()
random.shuffle(rows)

# Обновляем базу данных с перемешанными данными
cursor.execute("DELETE FROM positions")
for row in rows:
    cursor.execute("INSERT INTO positions VALUES (?,?)", row)

# Сохраняем изменения и закрываем соединение
conn.commit()
conn.close()