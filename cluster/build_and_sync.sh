#!/bin/bash

# Directorio actual
DIR=$(pwd)

# Compilar todos los .c en el directorio
echo "🔨 Compilando programas..."
for file in *.c; do
    exe="${file%.c}"
    echo " -> $file -> $exe"
    mpicc -o "$exe" "$file" -lm 
    chmod +x "$exe"
done

# Copiar todos los ejecutables a nodo2 (ajusta IP/nodo si tienes más)
echo "📤 Enviando ejecutables a mugiwara..."
scp $(find . -maxdepth 1 -type f -perm -111) mpiuser@192.168.0.21:$DIR/
