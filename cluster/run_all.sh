#!/bin/bash
# Script para ejecutar todos los ejercicios MPI en cluster de 2 máquinas

# Cambia según tu hostfile
HOSTS="--hostfile $HOME/hosts.txt"

# OBLIGA A DOS
MAP="-map-by ppr:2:node"

echo "=== EJECUCIÓN DE PROGRAMAS MPI ==="

# 3_1: Histograma
echo -e "\n>>> Ejecutando 3_1 (Histograma con bins)"
mpirun -np 4 $HOSTS $MAP ./3_1 <<EOF
8
0
100
4
12 34 56 78 90 23 45 67
EOF

# 3_2: Aproximación de π con Monte Carlo
echo -e "\n>>> Ejecutando 3_2 (Monte Carlo Pi)"
mpirun -np 4 $HOSTS $MAP ./3_2 <<EOF
1000000
EOF

# 3_3: Broadcast simple
echo -e "\n>>> Ejecutando 3_3 (Broadcast)"
mpirun -np 4 $HOSTS $MAP ./3_3

# 3_3_1: Scatter/Gather
echo -e "\n>>> Ejecutando 3_3_1 (Scatter/Gather)"
mpirun -np 4 $HOSTS $MAP ./3_3_1

# 3_4: Suma de vectores
echo -e "\n>>> Ejecutando 3_4 (Suma de vectores)"
mpirun -np 4 $HOSTS $MAP ./3_4

# 3_5: Producto Matriz-Vector
echo -e "\n>>> Ejecutando 3_5 (Matriz-Vector)"
mpirun -np 4 $HOSTS $MAP ./3_5 <<EOF
12
EOF

# 3_6: Producto Matriz-Matriz
echo -e "\n>>> Ejecutando 3_6 (Matriz-Matriz)"
mpirun -np 4 $HOSTS $MAP ./3_6 <<EOF
12
EOF

# 3_7: Ping-pong
echo -e "\n>>> Ejecutando 3_7 (Ping-Pong)"
mpirun -np 2 $HOSTS $MAP ./3_7

# 3_8: Sort
echo -e "\n>>> Ejecutando 3_8 (Sort)"
mpirun -np 3 $HOSTS $MAP ./3_8 12

# 3_9:
echo -e "\n>>> Ejecutando 3_9"
mpirun -np 4 $HOSTS $MAP ./3_9 12
