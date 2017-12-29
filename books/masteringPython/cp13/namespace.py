# namespace, pipe can use to share data between process

import multiprocessing

manager = multiprocessing.Manager()
namespace = manager.Namespace()
namespace.spam = 123
namespace.eggs = 456
