#include <stdio.h>
#include <stdlib.h>

#define BUFSIZE 100
int main( int argc, char * argv )
{
  FILE * f = fopen( argv[1], "w" );
  int h = atoi( argv[1] );
  int w = atoi( argv[2] );
  
  for (int j=0; j<h*w; ++j)
  {
    
  }
  return 0;
}
