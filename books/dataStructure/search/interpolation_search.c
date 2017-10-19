/* interpolation */
int interpolation_search(int *a, int n, int key){

  int low, high, mid;
  low = 0;
  high = n-1;
  
  while(low <= high){
    mid = low + (key - a[low]) /  (a[high] - a[low]) * (high - low);
    if (key < a[mid]){
      high = mid -1;
    } else if (key > a[mid]){
      low = mid + 1;
    } else {
      return mid;
    }
  }
  return -1;
}
