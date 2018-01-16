#include <apop.h>
#include <string.h>
#include "utilities.h"


int main(){
    connect_mysql("mory", "mory2016", "datamodel");
    apop_data *data = apop_query_to_data("select * from surpasser_power");
    printf("size 1 is %lu\n", data->matrix->size1);
    printf("size 2 is %lu\n", data->matrix->size2);

    apop_data_show(data);
    apop_data_free(data);
}
