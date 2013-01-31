#include <stdlib.h>

#include "API.h"
#include "Type.h"
#include "Util.h"


int main(int argc, char** argv) {

   API::ParseArgs(argc, argv);

   API::Prepare();

   API::Draw();

   API::WriteTGA();

   return EXIT_SUCCESS;
}
