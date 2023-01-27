#include <stdio.h>
#include <cmath>

int main() {
	int levels = 10;
	int bucket_size = ceil(255.0 / levels);
	printf("Bucket Size: %d for levels %d\n", bucket_size, levels);
	int test = 210;
	printf("%f", test / float(bucket_size));
	int value = floor( test/ float(bucket_size)) * bucket_size;
	printf("Test: %d\n", test);
	printf("Value: %d\n", value);
}