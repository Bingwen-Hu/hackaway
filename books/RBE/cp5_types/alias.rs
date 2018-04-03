/** type statement can be used to give a new name to an existing type
 *  Type must have CamelCase names, or the compiler will raise a warning
 * 
 *  the main use of aliases is to reduce boilerplate:
 *      IoResult<T> = Result<T, IoError>
 */

type NanoSecond = u64;
type Inch = u64;

#[allow(non_camel_case_types)]
type u64_t = u64;

fn main() {
    let nanoseconds: NanoSecond = 5 as u64_t;
    let inches: Inch = 2 as u64_t;

    println!("{} nanoseconds + {} inches = {} unit?", 
             nanoseconds, inches, nanoseconds + inches);
}