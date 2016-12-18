#pragma once

#include "util.h"
#include <cstdint>
#include <array>
#include <string>
#include <type_traits>
#include <iostream>

namespace Chianti
{
    namespace Values
    {
        /**
         * This is a fixed sized array value.
         */
        template <typename T, size_t S>
        class ArrayValue {
        public:
            /*!
             * Default constructor
             */
            ArrayValue() {}

            /*!
             * Uses the array v as value.
             *
             * @param v The new value of the array
             */
            ArrayValue(const std::array<T, S> & v) : value(v)
            { }

            /*!
             * Creates a new array value where all entries can be manually specified.
             *
             * @param initializerList List of array values
             */
            ArrayValue(std::initializer_list<T> initializerList)
            {
                // Make sure that the correct number of elements has been specified
                ::Chianti::Util::assertMsg(initializerList.size() == S, "TODO");

                size_t i = 0;
                for (T v : initializerList)
                {
                    this->value[i++] = v;
                }
            }

            /*!
             * Returns the i-th entry of the array.
             *
             * @param i The index to return
             */
            T operator[](size_t i) const
            {
                ::Chianti::Util::assertMsg(i < S, "TODO");
                return this->value[i];
            }

        private:
            /*!
             * This is the value storage.
             */
            std::array<T, S> value;
        };

        /*!
         * Prints an array value
         */
        template <class T, size_t S>
        std::ostream & operator<<(std::ostream & os, const ArrayValue<T, S> & v)
        {
            os << '[';
            for (size_t i = 0; i < S; i++)
            {
                os << v[i];
                if (i < S - 1)
                {
                    os << ", ";
                }
            }
            os << ']';
            return os;
        };

        /*!
         * This is the base class for all composite values.
         */
        template <class... Ts>
        class CompositeValue {
        };

        /*!
         * This is a node in the composite value list
         */
        template<class T, class... Ts>
        class CompositeValue<T, Ts...> : public CompositeValue<Ts...> {
        public:
            /*!
             * Default constructor
             */
            CompositeValue() {}

            // Make sure that we can call the final constructor with all types
            using CompositeValue<Ts...>::CompositeValue;

            /*!
             * Creates a new instance of the CompositeValue class.
             *
             * @param v The value that shall be assigned to the node.
             */
            CompositeValue(const T & v) : value(v), isActive(true) {}

            /*!
             * This is the value that is stored here
             */
            T value;
            /*!
             * Indicates whether is the value that has been set
             */
            bool isActive = false;
        };

        /*!
         * This is a node in the composite value list.
         * This is a specialization for strings to make initialization of composite values with string literals work
         */
        template<class... Ts>
        class CompositeValue<std::string, Ts...> : public CompositeValue<Ts...> {
        public:
            /*!
             * Default constructor
             */
            CompositeValue() {}

            // Make sure that we can call the final constructor with all types
            using CompositeValue<Ts...>::CompositeValue;

            /*!
             * Creates a new instance of the CompositeValue class.
             *
             * @param v The value that shall be assigned to the node.
             */
            CompositeValue(const std::string & v) : value(v), isActive(true) {}

            /*!
             * Creates a new instance of the CompositeValue class.
             *
             * @param v The value that shall be assigned to the node.
             */
            CompositeValue(const char* v) : value(v), isActive(true) {}

            /*!
             * This is the value that is stored here
             */
            std::string value;
            /*!
             * Indicates whether is the value that has been set
             */
            bool isActive = false;
        };

        /*!
         * This is a node in the composite value list.
         * This is a specialization for array values to make initialization possible.
         */
        template<class T, size_t S, class... Ts>
        class CompositeValue<ArrayValue<T, S>, Ts...> : public CompositeValue<Ts...> {
        public:
            /*!
             * Default constructor
             */
            CompositeValue() {}

            // Make sure that we can call the final constructor with all types
            using CompositeValue<Ts...>::CompositeValue;

            /*!
             * Creates a new instance of the CompositeValue class.
             *
             * @param v The value that shall be assigned to the node.
             */
            CompositeValue(const ArrayValue<T, S> & v) : value(v), isActive(true) {}

            /*!
             * Creates a new instance of the CompositeValue class.
             *
             * @param v The value that shall be assigned to the node.
             */
            CompositeValue(const std::array<T, S> & v) : value(v), isActive(true) {}

            /*!
             * Creates a new instance of the CompositeValue class.
             *
             * @param v The value that shall be assigned to the node.
             */
            CompositeValue(std::initializer_list<T> v) : value(v), isActive(true) {}

            /*!
             * This is the value that is stored here
             */
            ArrayValue<T, S> value;
            /*!
             * Indicates whether is the value that has been set
             */
            bool isActive = false;
        };

        template <size_t, class>
        class CompositeValueType {};

        /*!
         * Holds the type of the k-th composite value chain.
         */
        template <size_t k, class T, class... Ts>
        class CompositeValueType<k, CompositeValue<T, Ts...>> {
        public:
            typedef typename CompositeValueType<k - 1, CompositeValue<Ts...>>::type type;
        };

        /*!
         * Holds the type of the k-th composite value chain.
         */
        template <class T, class... Ts>
        class CompositeValueType<0, CompositeValue<T, Ts...>> {
        public:
            typedef T type;
        };

        /*!
         * Returns the k-th value of a composite value.
         */
        template <size_t k, class T, class... Ts>
        inline typename std::enable_if<k == 0, T>::type get(const CompositeValue<T, Ts...> & v)
        {
            return v.value;
        };

        /*!
         * Returns the k-th value of a composite value.
         */
        template <size_t k, class T, class... Ts>
        inline typename std::enable_if<k != 0, typename CompositeValueType<k, CompositeValue<T, Ts...>>::type>::type get(const CompositeValue<T, Ts...> & v)
        {
            return get<k - 1, Ts...>(v);
        };

        /*!
         * Returns whether or not the k-th element of a composite value has been set (is active).
         *
         * @param v The composite value to consider
         * @return Whether or not the k-th value is active
         */
        template <size_t k, class... Ts>
        inline typename std::enable_if<k == 0, bool>::type isActive(const CompositeValue<Ts...> & v)
        {
            return v.isActive;
        };

        /*!
         * Returns whether or not the k-th element of a composite value has been set (is active).
         *
         * @param v The composite value to consider
         * @return Whether or not the k-th value is active
         */
        template <size_t k, class T, class... Ts>
        inline typename std::enable_if<k != 0, bool>::type isActive(const CompositeValue<T, Ts...> & v)
        {
            const CompositeValue<Ts...>& base = v;
            return isActive<k - 1>(base);
        };
    }
}