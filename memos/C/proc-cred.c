#define _GNU_SOURCE
#include <unistd.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
	uid_t ruid;
	uid_t euid;
	uid_t suid;

	getresuid(&ruid, &euid, &suid);	

	printf("real user id: %ld \n", (long)ruid);
	printf("effctive user id: %ld \n", (long)euid);
	printf("saved user id: %ld \n", (long)suid);

	seteuid(2000);
	getresuid(&ruid, &euid, &suid);	
	
	printf("real user id: %ld \n", (long)ruid);
	printf("effctive user id: %ld \n", (long)euid);
	printf("saved user id: %ld \n", (long)suid);
	return 0;
}
