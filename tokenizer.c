
// tokenizer_client.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <stdint.h>
#include <stdbool.h>

#define SERVER_PORT 65432
#define SERVER_IP "127.0.0.1"
#define BUF_SIZE 4096

void send_json_to_tokenizer(const char *json_str, char *response_buf) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        perror("Socket creation failed");
        exit(1);
    }

    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(SERVER_PORT);
    inet_pton(AF_INET, SERVER_IP, &server_addr.sin_addr);

    if (connect(sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("Connection to tokenizer server failed");
        close(sock);
        exit(1);
    }

    send(sock, json_str, strlen(json_str), 0);

    int len = recv(sock, response_buf, BUF_SIZE - 1, 0);
    if (len < 0) {
        perror("Receive failed");
        close(sock);
        exit(1);
    }
    response_buf[len] = '\0';

    close(sock);
}

int main() {
    char response[BUF_SIZE];

    // Encode example
    const char *encode_request = "{\"mode\": \"encode\", \"text\": \"hello world\"}";
    send_json_to_tokenizer(encode_request, response);
    printf("Encode Response: %s\n", response);

    // Decode example
    const char *decode_request = "{\"mode\": \"decode\", \"tokens\": [464, 3290, 837]}";
    send_json_to_tokenizer(decode_request, response);
    printf("Decode Response: %s\n", response);

    return 0;
}
