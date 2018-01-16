#include <apop.h>


int main(){
    apop_opts.db_engine = 'm';
    apop_data *data = apop_query_to_data("select * from surpasser");
}
