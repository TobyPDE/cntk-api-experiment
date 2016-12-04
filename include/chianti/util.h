#pragma once

#include <string>

namespace Chianti
{
    namespace Util
    {
        /**
         * Asserts a condition and throws an assertion exception is the condition is violated.
         *
         * @param condition The condition to assert
         * @param message The exception message that is used if the condition is not fulfilled
         */
        void assertMsg(bool condition, const std::string & message)
        {

        }
    }
}