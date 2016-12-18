#pragma once

#include <exception>
#include <iostream>

namespace Chianti
{
    namespace Exception
    {
        /**
         * This exception is thrown when an illegal argument has been provided.
         */
        class IllegalArgumentException : public std::exception {
        public:
            /**
             * Initializes a new instance of the IllegalArgumentException class.
             *
             * @param message The exception message.
             */
            IllegalArgumentException(const char* message) : message(message) {}

            /**
             * Returns the exception message
             *
             * @return The exception message
             */
            virtual const char* what() const throw()
            {
                return this->message;
            }

        private:
            /**
             * This is the exception message.
             */
            const char* message;
        };

        /**
         * This function terminates the program because a non-recoverable error occured.
         *
         * @param message The message to show to the user
         * @param exitCode The exit code
         */
        inline void terminate(const char* message, int exitCode)
        {
            std::cerr << "Illegal system state reached: " << message << std::endl;
            exit(exitCode);
        }

        /**
         * Asserts a condition on a parameter and throws an IllegalArgumentException if the condition is violated.
         *
         * @param condition The condition to assert
         * @param message The exception message
         */
        inline void assertArgument(bool condition, const char* message)
        {
            if (!condition)
            {
                throw IllegalArgumentException(message);
            }
        }
    }
}