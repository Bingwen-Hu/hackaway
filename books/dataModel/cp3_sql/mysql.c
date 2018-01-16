#include <apop.h>
#include <string.h>


int main(){
    apop_opts.verbose++;
    apop_opts.db_engine = 'm';
    strcpy(apop_opts.db_user, "mory");
    strcpy(apop_opts.db_pass, "mory2016");
    apop_db_open("datamodel");
    apop_data *data = apop_query_to_data("select * from surpasser_power");
    printf("size 1 is %lu\n", data->matrix->size1);
    printf("size 2 is %lu\n", data->matrix->size2);

    apop_data_show(data);
}
