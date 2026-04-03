# Touché23-ValueEval
DOI: https://doi.org/10.5281/zenodo.6814563
Huggingface: https://huggingface.co/datasets/webis/Touche23-ValueEval
Version: 2024-01-24

Dataset for [Touché / SemEval 2023 Task 4; ValueEval: Identification of Human Values behind Arguments](https://touche.webis.de/semeval23/touche23-web). Based on the original [Webis-ArgValues-22 dataset](https://doi.org/10.5281/zenodo.5657249) accompanying the paper [Identifying the Human Values behind Arguments](https://webis.de/publications.html#kiesel_2022b), published at ACL'22. The main dataset contains 8865 arguments. The supplementary dataset adds another 459 arguments for a total of 9324 annotated arguments.


## Value Taxonomy
The `value-categories.json` describes the 20 value categories of this task through examples. Format:
```
{
  "<value category>": {
    "<level 1 value>": [
      "<exemplary effect a corresponding argument might target>",
      ...
    ], ...
  }, ...
}
```
The level 1 values are not used for the 2023 task (except for the annotation), but are still listed here for some might find them useful for understanding the value categories. See our paper on [Identifying the Human Values behind Arguments](https://webis.de/publications.html#kiesel_2022b) for the complete taxonomy.


## Argument Corpus
The annotated corpus in tab-separated value format. Contains the following files for the different dataset splits:
- `arguments-<split>.tsv`: Each row corresponds to one argument
    - `Argument ID`: The unique identifier for the argument
    - `Conclusion`: Conclusion text of the argument
    - `Stance`: Stance of the `Premise` towards the `Conclusion`; one of "in favor of", "against"
    - `Premise`: Premise text of the argument
- `labels-<split>.tsv`: Each row corresponds to one argument
    - `Argument ID`: The unique identifier for the argument
    - Other: Each other column corresponds to one value category, with a 1 meaning that the argument resorts to the value category and a 0 that not
- `level1-labels-<split>.tsv`: The same as `labels-<split>.tsv` but for the 54 level 1 values of the taxonomy (used in human annotation). Though not used for the 2023 task (except for the annotation), participants can still use them in their approaches.

For the main purpose of the ValueEval shared task, `<split>` is one of:
  - training               Arguments for training approaches (61%)
  - validation             Arguments for validating (optimizing) approaches (21%)
  - test                   Arguments for testing approaches; labels were published after the competition (on 2023-04-29, 18%)

The distribution of argument sources is the same for training, validation, and test. Arguments with the same conclusion are always in the same split.

In addition, we provide the following supplementary datasets (by name for `<split>`) that contain different kinds of arguments. These are intended to test the robustness of approaches along with the main shared task evaluation:
  - validation-zhihu       Arguments from the recommendation and hotlist section of the Chinese question-answering website Zhihu, which teams can use for training or validating more robust approaches; these have been part of the original Webis-ArgValues-22 dataset
  - test-nahjalbalagha     Arguments from and based on the Nahj al-Balagha (see `meta-arguments-f` below)
  - test-nyt               Arguments from the New York Times; arguments have to be downloaded:

### Downloading the New York Times Data
Due to copyright reasons, we can not yet provide the arguments of test-nyt for direct download. However, you can download it yourself using [our nyt-downloader program](https://github.com/touche-webis-de/touche-code/tree/main/semeval23/human-value-detection/nyt-downloader).


## Argument Meta-information
- `meta-arguments-a.tsv`: Each row corresponds to one argument (IDs starting with `A`) from the [IBM-ArgQ-Rank-30kArgs](https://research.ibm.com/haifa/dept/vst/debating_data.shtml#Argument%20Quality)
    - `Argument ID`: The unique identifier for the argument
    - `WA`: the quality label according to the weighted-average scoring function
    - `MACE-P`: the quality label according to the MACE-P scoring function
    - `stance_WA`: the stance label according to the weighted-average scoring function
    - `stance_WA_conf`: the confidence in the stance label according to the weighted-average scoring function
- `meta-arguments-c.tsv`: Each row corresponds to one argument (IDs starting with `C`) from the Chinese question-answering website [Zhihu](https://www.zhihu.com)
    - `Argument ID`: The unique identifier for the argument
    - `Conclusion Chinese`: The original chinese conclusion statement
    - `Premise Chinese`: The original chinese premise statement
    - `URL`: Link to the original statement the argument was taken from
- `meta-arguments-d.tsv`: Each row corresponds to one argument (IDs starting with `D`) from [https://www.groupdiscussionideas.com](https://www.groupdiscussionideas.com)
    - `Argument ID`: The unique identifier for the argument
    - `URL`: Link to the topic the argument was taken from
- `meta-arguments-e.tsv`: Each row corresponds to one argument (IDs starting with `E`) from [the Conference for the Future of Europe](https://futureu.europa.eu)
    - `Argument ID`: The unique identifier for the argument
    - `URL`: Link to the comment the argument was taken from
- `meta-arguments-f.tsv`: Each row corresponds to one argument (IDs starting with `F`). This file contains information on the 279 arguments in `arguments-test-nahjalbalagha.tsv` and 1047 additional arguments that were not labeled so far. This data was contributed by the language.ml lab (Doratossadat, Omid, Mohammad, Ehsaneddin) [1].
    - `Argument ID`: The unique identifier for the argument
    - `Conclusion Farsi`: Conclusion text of the argument in Farsi
    - `Stance Farsi`: Stance of the `Premise` towards the `Conclusion`, in Farsi
    - `Premise Farsi`: Premise text of the argument in Farsi
    - `Conclusion English`: Conclusion text of the argument in English (translated from Farsi)
    - `Stance English`: Stance of the `Premise` towards the `Conclusion`; one of "in favor of", "against"
    - `Premise English`: Premise text of the argument in English (translated from Farsi)
    - `Source`: Source text of the argument; one of "Nahj al-Balagha" [2], "Ghurar al-Hikam wa Durar ak-Kalim" [3]; their Farsi translations were used
    - `Method`: How the premise was extracted from the source; one of "extracted" (directly taken), "deduced"; the conclusion are deduced
- `meta-arguments-g.tsv`: Each row corresponds to one argument (IDs starting with `G`) from the [New York Times](https://www.nytimes.com)
    - `Argument ID`: The unique identifier for the argument
    - `URL`: Link to the article the argument was taken from
    - `Internet Archive timestamp`: Timestamp of the article's version in the Internet Archive that was used

[1] https://language.ml
[2] https://en.wikipedia.org/wiki/Nahj_al-Balagha
[3] https://en.wikipedia.org/wiki/Ghurar_al-Hikam_wa_Durar_al-Kalim


## Original Annotations
- `annotations-level1.tsv`: The original annotations by the crowdworkers (before MACE and curation)
    - `Argument ID`: The unique identifier for the argument
    - `Worker ID`: The unique identifier for the crowdworker (anonymized)
    - `Value`: The level 1 that was annotated
    - `Annotation`: The annotation by that worker for that argument and value, with a 1 meaning that the argument resorts to the value and a 0 that not
- `comments.level1.tsv`: The comments of the crowdworkers during annotation
    - `Argument ID`: The unique identifier for the argument
    - `Worker ID`: The unique identifier for the crowdworker (anonymized)
    - `Value`: The level 1 that was annotated, or "-" for a comment on the argument
    - `Comment`: The comment by that worker for that argument and value (or argument)


## Authors
- Johannes Kiesel, Bauhaus-Universität Weimar, johannes.kiesel@uni-weimar.de
- Nailia Mirzakhmedova, Bauhaus-Universität Weimar, nailia.mirzakhmedova@uni-weimar.de 
- Milad Alshomary, Paderborn University, milad.alshomary@upb.de
- Maximilian Heinrich, Bauhaus-Universität Weimar, maximilian.heinrich@uni-weimar.de
- Nicolas Handke, Universität Leipzig, n.handke@studserv.uni-leipzig.de
- Xiaoni Cai, Technische Universität München, caix@in.tum.de
- Valentin Barriere, CENIA, valentin.barriere@cenia.cl
- Doratossadat Dastgheib, Shahid Beheshti University, d_dastgheib@sbu.ac.ir
- Omid Ghahroodi, Sharif University of Technology, omid.ghahroodi98@sharif.edu
- Mohammad Ali Sadraei, Sharif University of Technology, m.sadraei@sharif.edu
- Ehsaneddin Asgari, UC Berkeley, asgari@berkeley.edu
- Lea Kawaletz, Heinrich-Heine-Universität Düsseldorf, lea.kawaletz@hhu.de
- Henning Wachsmuth, Paderborn University, henningw@upb.de
- Benno Stein, Bauhaus-Universität Weimar, benno.stein@uni-weimar.de


## Version History
- 2024-01-24
  - Fixed IDs of annotations and comments

- 2023-08-15
  - Added original annotations

- 2023-04-29
  - Added test set labels

- 2023-01-19
  - Added test-nyt (meta files and description)
  - Added meta files

- 2023-01-04
  - Added test-nahjalbalagha
  - Fix spelling of "in favor of" for arguments from group discussions

- 2022-12-05
  - Complete data for the ValueEval shared task

- 2022-07-11
  - Exchanged the values.json from original dataset with task-specific value-categories.json

- 2022-07-09
  - Initial


## License
This dataset is distributed under [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/).

