extern "C"
{
  void deformation_change_uchar_matrix(unsigned char *defor_arr, unsigned char *img_arr, int *matrix_h, int *matrix_v, int rows, int columns) {
    int i, j;
    for (i = 0; i < rows; i++) {
      for (j = 0; j < columns; j++) {
        int new_i = i + int(matrix_v[i * columns + j]);
        int new_j = j + int(matrix_h[i * columns + j]);
        if (new_i >= 0 && new_j >= 0 && new_i < rows && new_j < columns) {
          defor_arr[new_i * columns + new_j] = img_arr[i * columns + j];
        }
      }
    }
  }
}
