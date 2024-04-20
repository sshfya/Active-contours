# Active-contours
Программа, реализующая метод сегментации изображения с помощью активных контуров.

main.py (input_image) (initial_snake) (output_image) (alpha) (beta) (tau) (w_line) (w_edge) (kappa)

Аргументы:
input_image	 	Имя файла — входное изображение
initial_snake	 	Имя файла с начальным приближением для активного контура
output_image	 	Имя файла — выходное изображение
alpha	 	Параметр alpha внутренней энергии, отвечающий за растяжимость контура
beta	 	Параметр beta внутренней энергии, отвечающий за жесткость контура
tau	 	Шаг градиентного спуска
w_line	 	Вес слагаемого интенсивности во внешней энергии
w_edge	 	Вес слагаемого границ во внешней энергии
kappa	 	Коэффициент при нормализованной внешней силе
