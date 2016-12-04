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
        /*!
         * Abstract class for initialized atomic values.
         */
        template <class T>
        class AbstractAtomicValue {
        public:
            /*!
             * Default constructor that initializes the value.
             */
            AbstractAtomicValue() : value(0) {}

            /*!
             * Constructor that allows a user-defined initialization.
             *
             * @param x The initial value
             */
            AbstractAtomicValue(T x) : value(x) {}

            /*!
             * Converts the wrapper to a base value.
             *
             * @return The value of the literal.
             */
            operator T() const
            {
                return value;
            }

        protected:
            /*!
             * The actual value.
             */
            T value;
        };

        /**
         * Represents an initialized 64 bit integer value
         */
        class Int64Value : public AbstractAtomicValue<int64_t> {
        public:
            using AbstractAtomicValue::AbstractAtomicValue;
        };

        /**
         * Represents an initialized unsigned 64 bit integer value
         */
        class UInt64Value : public AbstractAtomicValue<uint64_t> {
        public:
            using AbstractAtomicValue::AbstractAtomicValue;
        };

        /**
         * Represents an initialized 32 bit integer value
         */
        class Int32Value : public AbstractAtomicValue<int32_t> {
        public:
            using AbstractAtomicValue::AbstractAtomicValue;
        };

        /**
         * Represents an initialized unsigned 32 bit integer value
         */
        class UInt32Value : public AbstractAtomicValue<uint32_t> {
        public:
            using AbstractAtomicValue::AbstractAtomicValue;
        };

        /**
         * Represents an initialized 32 bit floating point value
         */
        class Float32Value : public AbstractAtomicValue<float> {
        public:
            using AbstractAtomicValue::AbstractAtomicValue;
        };

        /**
         * Represents an initialized 64 bit floating point value
         */
        class Float64Value : public AbstractAtomicValue<double> {
        public:
            using AbstractAtomicValue::AbstractAtomicValue;
        };

        /**
         * Represents a boolean value that is initialized to false
         */
        class BoolValue : public AbstractAtomicValue<bool> {
        public:
            using AbstractAtomicValue::AbstractAtomicValue;
        };

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
             * Assigns the value v to all entries of the array.
             *
             * @param v The value of every entry of the array.
             */
            ArrayValue(T v)
            {
                for (size_t i = 0; i < S; i++)
                {
                    this->value[i] = v;
                }
            }

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
            ArrayValue(std::initializer_list<size_t> initializerList)
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
        typename std::enable_if<k == 0, T>::type get(const CompositeValue<T, Ts...> & v)
        {
            return v.value;
        };

        /*!
         * Returns the k-th value of a composite value.
         */
        template <size_t k, class T, class... Ts>
        typename std::enable_if<k != 0, typename CompositeValueType<k, CompositeValue<T, Ts...>>::type>::type get(const CompositeValue<T, Ts...> & v)
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
        typename std::enable_if<k == 0, bool>::type isActive(const CompositeValue<Ts...> & v)
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
        typename std::enable_if<k != 0, bool>::type isActive(const CompositeValue<T, Ts...> & v)
        {
            const CompositeValue<Ts...>& base = v;
            return isActive<k - 1>(base);
        };
    }
}