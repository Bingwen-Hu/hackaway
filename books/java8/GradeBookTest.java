public class GradeBookTest
{
    public static void main(String[] args)
    {
        int[][] gradesArray = {{87, 96, 70},
                               {68, 87, 90}, 
                               {100, 85, 97}, 
                               {83, 84, 97},
                               {89, 98, 87}};

        GradeBook myGradeBook = new GradeBook("CS101 Introduction to Java Programming",
                                              gradesArray);
        System.out.printf("Welcome to the grade book for %n%s%n%n",
                          myGradeBook.getCourseName());
        myGradeBook.processGrades();
    }
}
