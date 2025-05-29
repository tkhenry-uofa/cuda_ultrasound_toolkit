#include "test_app.h"

void
run_test_app()
{
	TestApp app;
	app.run();
	app.cleanup();
}


int main()
{
	// bool result = false;
	// result = readi_beamform();

	run_test_app();

	return 0;
}