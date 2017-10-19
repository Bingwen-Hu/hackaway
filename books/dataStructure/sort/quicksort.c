void quicksort(List *L){
  qsort(L, 1, L->length);
}

void qsort(List *L, int low, int high){
  int pivot;
  if (low < high){
    pivot  = Partition(L, low, high);

    qsort(L, low, pivot-1);
    qsort(L, pivot+1, high);
  }
}


int Partition(List *L, int low, int high){

  int pivotkey = L->r[low];
  
  while (low < high){
    while (low < high && L->r[high] >= pivotkey){
      high--;
    }
    swap(L, low, high);
    while (low < high && L->r[low] <= pivotkey){
      low++;
    }
    swap(L, low, high);
  }
  return low;
}


/* Note:
   
   In fact, I am thinking why everything I do will not have a happy ending?

*/
