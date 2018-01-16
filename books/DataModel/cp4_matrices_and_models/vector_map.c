#include <apop.h>
#include "utilities.h"

double waken(double power){
    return 5.0;
}

int main(){
    connect_mysql("mory", "mory2016", "datamodel"); 
    apop_data *d = apop_query_to_data(
        "select * from surpasser_power");
    Apop_col_t(d, "power_value", power_vector);
    // warning
    d->vector = apop_vector_map(power_vector, waken);
    apop_name_add(d->names, "waken", 'v');
    apop_data_show(d);

}
