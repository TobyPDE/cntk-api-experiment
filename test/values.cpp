#include <gtest/gtest.h>

#include "chianti/chianti.h"

TEST(values, compositeValue_initialize_list)
{
    // Arrange
    // Act
    Chianti::Values::CompositeValue<::Chianti::Values::ArrayValue<uint64_t, 2>, std::string> v({1, 2});

    // Assert
    ASSERT_TRUE(Chianti::Values::isActive<0>(v));
    ASSERT_FALSE(Chianti::Values::isActive<1>(v));
    ASSERT_EQ(1, Chianti::Values::get<0>(v)[0]);
    ASSERT_EQ(2, Chianti::Values::get<0>(v)[1]);
}

TEST(values, compositeValue_initialize_array)
{
    // Arrange
    // Act
    Chianti::Values::CompositeValue<::Chianti::Values::ArrayValue<uint64_t, 2>, std::string> v(std::array<uint64_t, 2>{1, 2});

    // Assert
    ASSERT_TRUE(Chianti::Values::isActive<0>(v));
    ASSERT_FALSE(Chianti::Values::isActive<1>(v));
    ASSERT_EQ(1, Chianti::Values::get<0>(v)[0]);
    ASSERT_EQ(2, Chianti::Values::get<0>(v)[1]);
}

TEST(values, compositeValue_initialize_string_literal)
{
    // Arrange
    // Act
    Chianti::Values::CompositeValue<::Chianti::Values::ArrayValue<uint64_t, 2>, std::string> v("foo");

    // Assert
    ASSERT_FALSE(Chianti::Values::isActive<0>(v));
    ASSERT_TRUE(Chianti::Values::isActive<1>(v));
    ASSERT_EQ("foo", Chianti::Values::get<1>(v));
}

TEST(values, compositeValue_initialize_string)
{
    // Arrange
    // Act
    Chianti::Values::CompositeValue<::Chianti::Values::ArrayValue<uint64_t, 2>, std::string> v(std::string("foo"));

    // Assert
    ASSERT_FALSE(Chianti::Values::isActive<0>(v));
    ASSERT_TRUE(Chianti::Values::isActive<1>(v));
    ASSERT_EQ("foo", Chianti::Values::get<1>(v));
}
