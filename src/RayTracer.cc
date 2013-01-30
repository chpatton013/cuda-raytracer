#include <stdlib.h>

#include "API.h"
#include "Type.h"
#include "Util.h"


void cudaWrapper();

int main(int argc, char** argv) {

   API::ParseArgs(argc, argv);

   API::Prepare();

   API::Draw();

   API::WriteTGA();
   
   cudaWrapper();

   return EXIT_SUCCESS;
}
