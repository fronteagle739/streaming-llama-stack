#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#include "../ds4_server.c"
int main(void) {
    test_direction_request_binding_validation();
    test_direction_quality_request_mapping();
    test_quality_speed_request_validation();
    puts("Direction request validation and four-phase quality transport PASS");
    return 0;
}
