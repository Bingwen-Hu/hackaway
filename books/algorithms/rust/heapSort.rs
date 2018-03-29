// heapSort in Rust
// reference to C/heapSort.c

#[derive(Debug)]
 struct Heap {
    data: [u32; 10],
    length: usize,
    heapsize: usize,
}


fn parent(i: usize) -> usize {
    (i - 1) / 2
}

fn left(i: usize) -> usize {
    2 * i + 1
}

fn right(i: usize) -> usize {
    2 * (i + 1)
}

fn swap(heap: &mut Heap, i: usize, j: usize) {
    let t = heap.data[i];
    heap.data[i] = heap.data[j];
    heap.data[j] = t;
}

fn max_heapify(mut heap: &mut Heap, i: usize) {
    let l = left(i);
    let r = right(i);
    let mut largest: usize;
    if l < heap.heapsize && heap.data[l] > heap.data[i] {
        largest = l;
    } else {
        largest = i;
    }

    if r < heap.heapsize && heap.data[r] > heap.data[i] {
        largest = r;
    }
    if largest != i {
        swap(&mut heap, i, largest);
        max_heapify(&mut heap, largest);
    }
}

fn build_max_heap(mut heap: &mut Heap) {
    heap.heapsize = heap.length;
    for i in (0 .. heap.length/2-1).rev() {
        max_heapify(&mut heap, i);
    }
}


fn main() {
    let mut heap = Heap {
        data: [4, 1, 3, 2, 16, 9, 10, 14, 8, 7],
        length: 10,
        heapsize: 0,
    };
    println!("Heap {:?}", heap);
    build_max_heap(&mut heap);
    println!("Heap {:?}", heap);
}